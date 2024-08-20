import os
import json
from PIL import Image


# 转换为coco格式的标注
def convert_to_coco_format(image_dir, annotations_file, output_file):
    with open(annotations_file, 'r', encoding='utf-8') as f:
        annotations = json.load(f)

    images = []
    anns = []
    categories = [{"id":1, "name":"object"}]
    image_id = 0
    annotation_id = 0

    for item in annotations:
        img_file = item["image_name"]
        caption = item["result"]

        # 构建图像的完整路径
        img_path = os.path.join(image_dir, img_file.replace("dataset/Val/", ""))
        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}")
            continue

        # 获取图像尺寸
        img = Image.open(img_path)
        width, height = img.size

        # 添加图像信息
        images.append({
            "id":image_id,
            "file_name":img_file,
            "height":height,
            "width":width
        })

        # 添加注释信息
        anns.append({
            "id":annotation_id,
            "image_id":image_id,
            "caption":caption,
            "category_id":1
        })
        annotation_id += 1
        image_id += 1

    # 生成COCO格式的JSON文件
    coco_format = {
        "images":images,
        "annotations":anns,
        "categories":categories
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(coco_format, f, indent=4)


image_dir = r'E:\nlp\final\dataset\Val'  # 图像文件所在的根目录
annotations_file = r'E:\nlp\final\CNN-RNN\output_val_simple.json'
output_file = r'E:\nlp\final\CNN-RNN\coco_annotations_val_simple.json'
convert_to_coco_format(image_dir, annotations_file, output_file)
