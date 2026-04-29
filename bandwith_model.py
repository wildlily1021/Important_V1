import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, LeakyReLU
from keras.optimizers import RMSprop
from keras.applications import ResNet101
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import random
import matplotlib.pyplot as plt

# 定义图像大小
IMG_HEIGHT = 128
IMG_WIDTH = 128

# 数据集路径
data_dir = './dataset_10G'

# 用于存储图像和标签的列表
images = []
labels = []

# 遍历数据集文件夹
for folder_name in os.listdir(data_dir):
    folder_path = os.path.join(data_dir, folder_name)
    if os.path.isdir(folder_path):
        label = float(folder_name)  # 文件夹名称即为标签
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
            images.append(img)
            labels.append(label)

# 转换为NumPy数组
images = np.array(images)
labels = np.array(labels)
# 归一化图像数据
images = images / 255.0

# 随机打乱数组索引
indices = list(range(len(images)))
indices_2 = list(range(len(labels)))
random.shuffle(indices)

# 根据打乱后的索引重新排列数组
shuffled_images = images[indices]
shuffled_labels = labels[indices]

# 归一化labels
min_label = np.min(labels, axis=0)
max_label = np.max(labels, axis=0)

# 进行 Min-Max 归一化
shuffled_labels = (shuffled_labels - min_label) / (max_label - min_label)

# 将数据集分为训练集、验证集和测试集
X_train, X_temp, y_train, y_temp = train_test_split(shuffled_images, shuffled_labels, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# 数据增强
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
datagen.fit(X_train)

# # 使用ResNet50作为基础模型
# base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
# 使用ResNet101作为基础模型
base_model = ResNet101(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))

# # 冻结ResNet50的卷积层
# for layer in base_model.layers:
#     layer.trainable = False

# 解冻ResNet101的最后几个卷积层
for layer in base_model.layers[:-30]:
    layer.trainable = False

for layer in base_model.layers[-30:]:
    layer.trainable = True

# # 构建模型
# x = base_model.output
# x = GlobalAveragePooling2D()(x)
# # x = Dense(1024, activation='relu')(x)  # 去掉L2正则化
# x = Dense(1024, activation='relu', kernel_initializer=HeNormal())(x)  # 使用He初始化
# x = Dropout(0.3)(x)
# predictions = Dense(1)(x)

# 构建模型
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, kernel_initializer='he_normal', kernel_regularizer=l2(0.001))(x)  # 轻量L2正则化
x = LeakyReLU(alpha=0.1)(x)
x = Dropout(0.3)(x)
predictions = Dense(1)(x)

# 定义最终模型
model = Model(inputs=base_model.input, outputs=predictions)

# 使用RMSprop优化器编译模型
model.compile(optimizer=RMSprop(learning_rate=0.0001), loss='mean_squared_error')

# 定义回调函数
# early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00000001)  # 以前是0.00001

# 训练模型
history = model.fit(X_train, y_train, epochs=150, validation_data=(X_val, y_val),
          callbacks=[reduce_lr], batch_size=16)

# 保存模型
model.save('image_regression_model_rmsprop_Hestart.h5')

# 测试模型
model = tf.keras.models.load_model('image_regression_model_rmsprop_Hestart.h5')
predictions = model.predict(X_test)

# 反归一化预测值
predictions_original = predictions * (max_label - min_label) + min_label

# 打印实际值与预测值的对比
for i in range(20):  # 随机打印10个测试样本的结果
    print(f"实际值: {y_test[i] * (max_label - min_label) + min_label}, 预测值: {predictions_original[i][0]}")

# 计算均方误差（MSE）作为模型的评估指标
mse = np.mean((y_test - predictions.flatten()) ** 2)
print(f"测试集的均方误差（MSE）: {mse}")

# 绘制训练曲线
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()

