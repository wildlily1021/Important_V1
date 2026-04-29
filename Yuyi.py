# import os
# import json
# import numpy as np
# from PIL import Image
#
# def compress_image(image_path, output_dir, target_width=640):
#     """
#     压缩图像并保持原比例，同时保留原有文件名。
#
#     参数:
#     - image_path: 原始图像路径
#     - output_dir: 压缩后的图像保存目录
#     - target_width: 压缩后的宽度，默认640像素
#     """
#     # 获取原始文件名
#     filename = os.path.basename(image_path)
#
#     output_path = os.path.join(output_dir, filename)
#
#     with Image.open(image_path) as img:
#         # 计算保持比例的高度
#         width_percent = target_width / float(img.size[0])
#         target_height = int((float(img.size[1]) * float(width_percent)))
#
#         # 压缩图像，使用LANCZOS滤镜保持高质量
#         img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
#         img.save(output_path)
#         print(f"图像压缩成功并保存为：{output_path}")
#
#     return output_path
#
#
# # 原始图片目录
# input_folder = "./dataset_YOLO"
#
# output_dir = "./dataset_YOLO/images"
# # COCO格式标注输出路径
# output_json_path = "./dataset_YOLO/annotations/coco_annotations.json"
#
# # 初始化COCO格式
# coco_format = {
#     "images": [],
#     "annotations": [],
#     "categories": [
#         {"id": 0, "name": "red_region"}  # 类别0，表示深红色区域
#     ]
# }
#
# annotation_id = 0
# image_id = 0
# target_width = 320  # 压缩后的宽度
#
# # 循环遍历
# for i in np.arange(0.5, 2.75, 0.25):
#     for j in range(1, 11):
#         for k in range(1, 101):
#             for Modulation in range(1, 5):
#                 # 调用函数，计算相关参数
#                 Fs = 10 * 1e9
#                 Fs_index = Fs / 1e9
#                 Rs_index = i
#                 Fc = 3.0 + j * 0.05 - 0.05
#                 SNR = -10 + (k - 1) * 0.2
#                 row_num = 4 * 100 * 10 * ((i - 0.5) / 0.25) + (j - 1) * 100 * 4 + (k - 1) * 4 + Modulation
#                 image_id = int(row_num - 1)
#                 Fc_index = round(Fc, 2)
#                 SNR_index = round(SNR, 1)
#
#                 # 构建原始图片路径
#                 original_image_path = f"{input_folder}/image_{image_id}.jpg"
#
#                 # 新的文件名按照 image_序号.jpg 格式命名
#                 new_image_name = f"image_{image_id}.jpg"
#                 new_image_path = os.path.join(input_folder, new_image_name)
#                 compress_image(original_image_path, output_dir)
#                 # 将原图片重命名
#                 if os.path.exists(original_image_path):
#                     os.rename(original_image_path, new_image_path)
#
#                     x_min = 0.1
#                     x_max = 3720 / 640
#                     y_max = (10 - (Fc_index - 0.5 * Rs_index)) / 10 * 3720 / 640
#                     y_min = (10 - (Fc_index + 0.5 * Rs_index)) / 10 * 3720 / 640
#
#                     # 生成分割信息（根据你自己的逻辑来计算轮廓）
#                     width = x_max - x_min  # 示例
#                     height = y_max - y_min  # 示例
#
#                     # 添加图像元数据到COCO格式
#                     coco_format["images"].append({
#                         "id": image_id,
#                         "file_name": f"image_{image_id}.jpg",
#                         "height": int(y_min + height),  # 假设图像的高度基于红色区域
#                         "width": int(x_min + width)  # 假设图像的宽度基于红色区域
#                     })
#
#                     # 生成分割掩码（polygon格式）
#                     segmentation = [
#                         [x_min, y_min, x_min + width, y_min, x_min + width, y_min + height, x_min, y_min + height]
#                     ]
#
#                     # 生成边界框
#                     bbox = [x_min, y_min, width, height]
#
#                     # 添加标注信息
#                     coco_format["annotations"].append({
#                         "id": annotation_id + 1,
#                         "image_id": (image_id + 1),
#                         "category_id": 0,  # 深红色区域类别
#                         "segmentation": segmentation,
#                         "area": width * height,
#                         "bbox": bbox,
#                         "iscrowd": 0
#                     })
#
#                     # 更新ID
#                     annotation_id += 1
#                     # image_id += 1
#
# # 将COCO标注数据保存为JSON格式
# with open(output_json_path, 'w') as json_file:
#     json.dump(coco_format, json_file, indent=4)
#
# print(f"COCO格式标注数据保存至 {output_json_path}")

# import os
# import json
# import random
# import shutil

# def split_dataset(
#     images_dir, annotation_file, output_dir, split_ratio=0.8, seed=42
# ):
#     # 设置随机种子保证可重复性
#     random.seed(seed)
#
#     # 创建输出目录
#     train_images_dir = os.path.join(output_dir, "train")
#     val_images_dir = os.path.join(output_dir, "val")
#     os.makedirs(train_images_dir, exist_ok=True)
#     os.makedirs(val_images_dir, exist_ok=True)
#
#     # 读取COCO标注文件
#     with open(annotation_file) as f:
#         coco_data = json.load(f)
#
#     images = coco_data["images"]
#     annotations = coco_data["annotations"]
#
#     # 随机划分图像为训练集和验证集
#     random.shuffle(images)
#     split_index = int(len(images) * split_ratio)
#     train_images = images[:split_index]
#     val_images = images[split_index:]
#
#     # 根据划分的图像提取相应的标注
#     def filter_annotations(images):
#         image_ids = {img["id"] for img in images}
#         return [ann for ann in annotations if ann["image_id"] in image_ids]
#
#     train_annotations = filter_annotations(train_images)
#     val_annotations = filter_annotations(val_images)
#
#     # 创建新的COCO格式JSON数据
#     def create_coco_data(images, annotations):
#         return {
#             "images": images,
#             "annotations": annotations,
#             "categories": coco_data["categories"],
#         }
#
#     train_coco = create_coco_data(train_images, train_annotations)
#     val_coco = create_coco_data(val_images, val_annotations)
#
#     # 保存新的JSON文件
#     with open(os.path.join(output_dir, "instances_train.json"), "w") as f:
#         json.dump(train_coco, f)
#
#     with open(os.path.join(output_dir, "instances_val.json"), "w") as f:
#         json.dump(val_coco, f)
#
#     # 将图像文件复制到对应的目录
#     def copy_images(images, target_dir):
#         for img in images:
#             src_path = os.path.join(images_dir, img["file_name"])
#             dst_path = os.path.join(target_dir, img["file_name"])
#             shutil.copy(src_path, dst_path)
#
#     copy_images(train_images, train_images_dir)
#     copy_images(val_images, val_images_dir)
#
#     print("数据集划分完成！")
#     print(f"训练集图像数: {len(train_images)}, 验证集图像数: {len(val_images)}")
#
#
# # 使用示例
# split_dataset(
#     images_dir="./dataset_YOLO/images",
#     annotation_file="./dataset_YOLO/annotations/coco_annotations.json",
#     output_dir="./dataset_YOLO/val",
#     split_ratio=0.8,
#     seed=42
# )

import os
import json
import random
import shutil

# 配置路径
image_folder = "./dataset_YOLO/images"
annotation_path = "./dataset_YOLO/annotations/coco_annotations.json"

train_image_folder = "./dataset_YOLO/train"
val_image_folder = "./dataset_YOLO/val"

train_json_path = "./dataset_YOLO/annotations/train.json"
val_json_path = "./dataset_YOLO/annotations/val.json"

# 创建训练和验证文件夹
os.makedirs(train_image_folder, exist_ok=True)
os.makedirs(val_image_folder, exist_ok=True)

# 读取原始COCO标注文件
with open(annotation_path, 'r') as f:
    coco_data = json.load(f)

# 初始化训练和测试集的COCO格式结构
train_data = {
    "images": [],
    "annotations": [],
    "categories": coco_data["categories"]
}
val_data = {
    "images": [],
    "annotations": [],
    "categories": coco_data["categories"]
}

# 打乱并划分数据：80%训练，20%测试
all_images = coco_data["images"]
random.shuffle(all_images)
split_index = int(0.8 * len(all_images))

train_images = all_images[:split_index]
val_images = all_images[split_index:]

# 构建image_id到annotations的映射表
image_id_to_annotations = {}
for ann in coco_data["annotations"]:
    image_id = ann["image_id"]
    if image_id not in image_id_to_annotations:
        image_id_to_annotations[image_id] = []
    image_id_to_annotations[image_id].append(ann)

# 填充数据并移动图像文件
def populate_dataset(dataset, images, image_folder, output_folder):
    for img in images:
        SHAP_combined dataset["images"].append(img)

        # 找到对应标注，并加入数据集
        anns = image_id_to_annotations.get(img["id"], [])
        dataset["annotations"].extend(anns)

        # 移动对应的图像文件到新的文件夹
        src_path = os.path.join(image_folder, img["file_name"])
        dest_path = os.path.join(output_folder, img["file_name"])
        if os.path.exists(src_path):
            shutil.copyfile(src_path, dest_path)
        else:
            print(f"警告：找不到文件 {src_path}")

# 填充训练集和测试集
populate_dataset(train_data, train_images, image_folder, train_image_folder)
populate_dataset(val_data, val_images, image_folder, val_image_folder)

# 保存训练集和测试集的JSON标注文件
with open(train_json_path, 'w') as f:
    json.dump(train_data, f, indent=4)

with open(val_json_path, 'w') as f:
    json.dump(val_data, f, indent=4)

print("数据集划分和图像移动完成：")
print(f"训练集图片保存至：{train_image_folder}")
print(f"测试集图片保存至：{val_image_folder}")
print(f"训练集标注保存至：{train_json_path}")
print(f"测试集标注保存至：{val_json_path}")
#
