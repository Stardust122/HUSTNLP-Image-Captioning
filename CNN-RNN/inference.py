import json
from PIL import Image
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from data_loader import get_loader
from torchvision import transforms
from model import EncoderCNN, DecoderRNN

encoder_file = 'encoder-3.pkl'
decoder_file = 'decoder-3.pkl'

embed_size = 256
hidden_size = 512

transform_test = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])


data_loader = get_loader(transform=transform_test, mode='test')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vocab_size = len(data_loader.dataset.vocab)

encoder = EncoderCNN(embed_size)
decoder = DecoderRNN(embed_size, hidden_size, vocab_size)
encoder.eval()
decoder.eval()
encoder.load_state_dict(torch.load(os.path.join('E:/nlp/final/CNN-RNN/models', encoder_file)))
decoder.load_state_dict(torch.load(os.path.join('E:/nlp/final/CNN-RNN/models', decoder_file)))
encoder.to(device)
decoder.to(device)


def clean_sentence(output):  # 用于将模型输出的单词序列转换为句子
    words_sequence = []

    for i in output:
        if i == 1:
            continue
        word = data_loader.dataset.vocab.idx2word[i]
        words_sequence.append(word)
        if word == '.':
            break

    words_sequence = words_sequence[1:]
    sentence = ' '.join(words_sequence)
    sentence = sentence.capitalize()

    return sentence


def get_prediction():  # 获取模型的预测结果并显示图像
    orig_image, image = next(iter(data_loader))
    image = image.to(device)
    features = encoder(image).unsqueeze(1)
    output = decoder.sample(features)
    sentence = clean_sentence(output)
    print(sentence)
    plt.imshow(np.squeeze(orig_image))
    plt.title('Sample Image')
    plt.show()


def load_image(image_path, transform):  # 加载和预处理图像
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    return image


def generate_predictions(image_folder, encoder, decoder, device, output_file='predictions.json'):  # 生成预测结果
    predictions = []

    # 遍历图像文件夹中的所有图像文件
    for image_filename in os.listdir(image_folder):
        image_path = os.path.join(image_folder, image_filename)

        # 加载并预处理图像
        image = load_image(image_path, transform_test)
        image = image.unsqueeze(0).to(device)  # 添加批次维度并移动到设备

        # 编码和解码图像
        features = encoder(image).unsqueeze(1)
        output = decoder.sample(features)
        sentence = clean_sentence(output)

        # 去掉文件扩展名以获取 ID
        image_id = int(os.path.splitext(image_filename)[0])

        # 将结果添加到列表中
        predictions.append({
            "image_id":image_id,
            "caption":sentence
        })

    # 将结果写入 JSON 文件
    with open(output_file, 'w') as f:
        json.dump(predictions, f, indent=4)

    print(f"Predictions saved to {output_file}")


# # 推理单张图片
# get_prediction()

# 推理全部图片并生成 JSON 文件
image_folder = r'E:\nlp\final\dataset\Val'
generate_predictions(image_folder, encoder, decoder, device)