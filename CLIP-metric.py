import torch
from transformers import CLIPProcessor, CLIPModel
import json
from PIL import Image
import os
import torch.nn.functional as F

# 加载预训练的CLIP模型和处理器
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# 读取图像描述的JSON文件
with open('.cache_fc_val.json', 'r', encoding='utf-8') as f:
    captions_data = json.load(f)

# 批量处理
results = []
total_score = 0.0
count = 0

def load_image(image_filename):
    # 根据图像文件名加载图像的函数，请确保路径和文件名正确
    image_path = f'nlp_val/{image_filename}.jpg'
    image = Image.open(image_path)
    return image

for item in captions_data:

    image_id = item['image_id']
    caption = item['caption']

    #trans
    image = load_image(image_id)
    #RNN+CNN
    #image_filename = f"{int(image_id) + 2002}"
    #image = load_image(image_filename)

    # 处理图像和文本
    inputs = processor(text=[caption], images=image, return_tensors="pt", padding=True)

    # 获取模型输出
    with torch.no_grad():
        outputs = model(**inputs)

    # 使用余弦相似度计算图像与描述的相似度分数，并规范化到 [-1, 1]
    cosine_similarity = F.cosine_similarity(outputs.image_embeds, outputs.text_embeds)
    score = cosine_similarity.item()

    # 保存结果
    results.append({
        "image_id": image_id,
        "caption": caption,
        "similarity_score": score
    })

    # 更新总分数和计数器
    total_score += score
    count += 1

# 保存所有评估结果到JSON文件
with open('trans_CLIP_l.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

# 计算并输出平均分数
average_score = total_score / count if count > 0 else 0
print(f"批量处理完成，结果已保存。所有图片的平均相似度分数为：{average_score:.4f}")
