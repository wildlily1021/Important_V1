import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# 定义图像大小
IMG_HEIGHT = 128
IMG_WIDTH = 128

# 数据集路径
data_dir = 'D:/postgraduate/AI_SIGNAL_ANA/pythonProject_qt/dataset'

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

# 将数据集分为训练集、验证集和测试集
X_train, X_temp, y_train, y_temp = train_test_split(images, labels, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# 构建卷积神经网络模型
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1)  # 输出层，预测数值
])

# 编译模型
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# 训练模型
model.fit(X_train, y_train, epochs=50, validation_data=(X_val, y_val), batch_size=32)

# 保存模型
model.save('image_regression_model.h5')

# 测试模型
model = tf.keras.models.load_model('image_regression_model.h5')
predictions = model.predict(X_test)

# 打印实际值与预测值的对比
for i in range(10):  # 随机打印10个测试样本的结果
    print(f"实际值: {y_test[i]}, 预测值: {predictions[i][0]}")

# 计算均方误差（MSE）作为模型的评估指标
mse = np.mean((y_test - predictions.flatten()) ** 2)
print(f"测试集的均方误差（MSE）: {mse}")

# predictions = model.predict(new_images)
