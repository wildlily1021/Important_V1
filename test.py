import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import rcParams
from tensorflow import keras
from keras.models import Model
from keras import backend as K
# from photo_save import photo_save
# from excelwrite import write_to_excel_signal
# from scipy.signal import hilbert
# from scipy.fft import fft
# from scipy.ndimage import uniform_filter1d

# 设置字体为 Calibri
rcParams['font.family'] = 'Calibri'
def grad_cam(input_model, image, conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        inputs=[input_model.inputs],
        outputs=[input_model.get_layer(conv_layer_name).output, input_model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image)
        if pred_index is None:
            pred_index = 0  # 对于回归问题，我们选择第一个输出
        output = predictions[:, pred_index]

    grads = tape.gradient(output, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

    return heatmap.numpy()


def display_gradcam(img_path, model, conv_layer_names, alpha=0.9):
    img = cv2.imread(img_path)
    IMG_HEIGHT = 128
    IMG_WIDTH = 128
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_array = img_rgb / 255.0
    img_input = np.expand_dims(img_array, axis=0)

    # 生成2x2网格图（左侧小图）
    plt.figure(figsize=(12, 12))

    for idx, conv_layer_name in enumerate(conv_layer_names):
        heatmap = grad_cam(model, img_input, conv_layer_name)
        heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        superimposed_img = heatmap_colored * alpha
        superimposed_img = np.uint8(superimposed_img)

        plt.subplot(2, 2, idx + 1, aspect=4/3)
        plt.title(f'Grad-CAM ({conv_layer_name})')
        plt.imshow(superimposed_img)
        plt.axis('off')

    plt.tight_layout()
    # 保存叠加图像
    output_path = f'./signal_ana/bandwidth_Grad_Cam.jpg'
    plt.savefig(output_path)
    plt.close()
    
    # 生成右侧两个大图（带坐标轴）
    # 选择两个最重要的层：conv5_block3_out 和 conv4_block6_out
    important_layers = [conv_layer_names[0], conv_layer_names[2]]  # conv5_block3_out 和 conv4_block6_out
    
    for idx, conv_layer_name in enumerate(important_layers):
        heatmap = grad_cam(model, img_input, conv_layer_name)
        heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        superimposed_img = heatmap_colored * alpha
        superimposed_img = np.uint8(superimposed_img)

        # 创建单独的大图，带坐标轴
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(superimposed_img, aspect='auto', extent=[0, IMG_WIDTH, 0, IMG_HEIGHT])
        ax.set_xlabel('Feature Space X', fontsize=14, color='white', fontweight='bold')
        ax.set_ylabel('Feature Space Y', fontsize=14, color='white', fontweight='bold')
        ax.set_title(f'Grad-CAM ({conv_layer_name})', fontsize=16, color='white', pad=15, fontweight='bold')
        
        # 设置坐标轴刻度（使用更合理的刻度间隔）
        num_ticks = 6
        x_ticks = np.linspace(0, IMG_WIDTH, num_ticks)
        y_ticks = np.linspace(0, IMG_HEIGHT, num_ticks)
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)
        ax.set_xticklabels([f'{int(x)}' for x in x_ticks], color='white', fontsize=12, fontweight='bold')
        ax.set_yticklabels([f'{int(y)}' for y in y_ticks], color='white', fontsize=12, fontweight='bold')
        
        # 设置背景为黑色
        ax.set_facecolor('black')
        fig.patch.set_facecolor('black')
        # 增强坐标轴可见性
        ax.tick_params(colors='white', which='both', width=2, length=6, labelsize=12)
        ax.spines['bottom'].set_color('white')
        ax.spines['top'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['right'].set_color('white')
        ax.spines['bottom'].set_linewidth(2)
        ax.spines['top'].set_linewidth(2)
        ax.spines['left'].set_linewidth(2)
        ax.spines['right'].set_linewidth(2)
        
        # 保存右侧大图（覆盖原来的bandwidth_Grad_Cam.jpg用于右侧上方显示）
        if idx == 0:
            # 第一个大图保存为右侧上方显示的文件
            output_path_large = f'./signal_ana/bandwidth_Grad_Cam_large1.jpg'
        else:
            # 第二个大图保存为右侧下方显示的备用文件（如果需要）
            output_path_large = f'./signal_ana/bandwidth_Grad_Cam_large2.jpg'
        plt.savefig(output_path_large, facecolor='black', bbox_inches='tight', pad_inches=0.1, dpi=150)
        plt.close(fig)
    # plt.show()
def bandwidth_predict(img_path, img_path_grad):
    min_label = 64.6833
    max_label = 355.7581

    # 定义图像大小
    IMG_HEIGHT = 128
    IMG_WIDTH = 128

    # 测试模型
    model = tf.keras.models.load_model('image_regression_model_rmsprop_Hestart.h5')

    # 载入图像
    img = cv2.imread(img_path)
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    images = np.array(img)
    new_images = (images / 255.0).reshape(1, IMG_WIDTH, IMG_HEIGHT, 3)
    # see = display_grad_cam(img_path, i, j, model, layer_name=last_conv_layer_name)  # 替换为您模型的最后一个卷积层的名称
    predictions = model.predict(new_images)

    # 反归一化预测值
    predictions_original = predictions * (max_label - min_label) + min_label
    predictions_float = float(predictions_original[0])

    # 选择几个不同的卷积层
    conv_layer_names = [
        'conv5_block3_out',  # 最后的卷积层
        'conv5_block1_out',  # 倒数第三层
        'conv4_block6_out',  # 倒数第二层的卷积层(good)
        'conv4_block23_out'  # 中间层的卷积层
    ]

    # 显示多个卷积层的Grad-CAM结果
    display_gradcam(img_path_grad, model, conv_layer_names)
    plt.clf()
    plt.close('all')
    return predictions_float






# last_conv_layer_name = get_last_conv_layer_name(model)
# display_gradcam(f'./signal_ana/STFT_Org.jpg', model, last_conv_layer_name)


    # for i in np.arange(1.0, 6, 0.5):
    #     for j in range(1, 21):
    #         img_path = f'./test/image{i}_{j}.jpg'
    #         img = cv2.imread(img_path)
    #         img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    #         images = np.array(img)
    #         new_images = (images / 255.0).reshape(1, IMG_WIDTH, IMG_HEIGHT, 3)
    #         # see = display_grad_cam(img_path, i, j, model, layer_name=last_conv_layer_name)  # 替换为您模型的最后一个卷积层的名称
    #         predictions = model.predict(new_images)
    #         # 反归一化预测值
    #         predictions_original = predictions * (max_label - min_label) + min_label
    #         predictions_float = float(predictions_original[0])
    #         center_frequency_output = process_image(i, j)
    #         images.append(img)
    #         row_num = ((i - 1) / 0.5) * 20 + j
    #         if j < 11:
    #             Fc = 12
    #             snr = -11 + j
    #         else:
    #             Fc = 13
    #             snr = j - 21
    #
    #         wb1 = openpyxl.load_workbook(file_path)
    #         sheet1 = wb1["Sheet1"]
    #
    #         # sheet = wb.active
    #         # Convert row_number to 0-based index for openpyxl
    #         row_index = int(row_num)
    #         # Write data to specified row and columns A, B, C
    #         sheet1.cell(row=row_index + 1, column=3).value = Fc
    #         sheet1.cell(row=row_index + 1, column=4).value = i
    #         sheet1.cell(row=row_index + 1, column=6).value = snr
    #         wb1.save(file_path)
    #
    #         wb2 = openpyxl.load_workbook(file_path)
    #         sheet2 = wb2["Sheet2"]
    #         sheet2.cell(row=row_index + 1, column=2).value = predictions_float
    #         sheet2.cell(row=row_index + 1, column=3).value = center_frequency_output
    #         wb2.save(file_path)
    #         print(f"Data successfully written to row {row_num}")



