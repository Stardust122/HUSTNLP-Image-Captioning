import torch
import torch.nn as nn
import torchvision.models as models


# CNN 编码器类，用于提取图像特征
class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        # 使用预训练的 ResNet50 模型
        resnet = models.resnet50(pretrained=True)
        # 冻结 ResNet 参数，不进行训练
        for param in resnet.parameters():
            param.requires_grad_(False)

        # 移除 ResNet 的最后一层全连接层
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        # 线性层将 ResNet 的输出特征转换为指定的嵌入维度
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    # 前向传播，输入图像，输出特征向量
    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)  # 扁平化特征
        features = self.embed(features)  # 映射到嵌入空间
        return features


# RNN 解码器类，用于生成图像描述
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()
        # 嵌入层：将单词映射到嵌入向量
        self.embedding_layer = nn.Embedding(vocab_size, embed_size)
        # LSTM 层：用于序列建模
        self.lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        # 线性层：将 LSTM 的输出映射回词汇表大小
        self.linear = nn.Linear(hidden_size, vocab_size)

    # 前向传播，输入特征和标签，输出预测结果
    def forward(self, features, captions):
        captions = captions[:, :-1]  # 去掉最后一个时间步的标签
        embed = self.embedding_layer(captions)  # 将标签嵌入
        embed = torch.cat((features.unsqueeze(1), embed), dim=1)  # 将图像特征拼接到序列前
        lstm_outputs, _ = self.lstm(embed)  # 输入 LSTM
        out = self.linear(lstm_outputs)  # 映射到词汇表
        return out

    # 采样方法，用于在预测时生成句子
    def sample(self, inputs, states=None, max_len=20):
        output_sentence = []
        for i in range(max_len):
            lstm_outputs, states = self.lstm(inputs, states)  # 通过 LSTM 生成下一个单词
            lstm_outputs = lstm_outputs.squeeze(1)
            out = self.linear(lstm_outputs)  # 映射到词汇表
            last_pick = out.max(1)[1]  # 选择概率最大的单词
            output_sentence.append(last_pick.item())  # 将单词添加到输出句子中
            inputs = self.embedding_layer(last_pick).unsqueeze(1)  # 准备下一个时间步的输入

        return output_sentence  # 返回生成的句子
