import torch
import sys
import torch.nn as nn
import torch.utils.data as data
import math
import numpy as np
import os
from torchvision import transforms
from data_loader import get_loader
from model import EncoderCNN, DecoderRNN

# 超参数设置
batch_size = 8
vocab_threshold = 1
vocab_from_file = True
embed_size = 256
hidden_size = 512
num_epochs = 3
save_every = 1
print_every = 32
log_file = 'training_log.txt'

# 图像预处理
transform_train = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

# 获取数据加载器
data_loader = get_loader(transform=transform_train,
                         mode='train',
                         batch_size=batch_size,
                         vocab_threshold=vocab_threshold,
                         vocab_from_file=vocab_from_file)

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 模型初始化
vocab_size = len(data_loader.dataset.vocab)
encoder = EncoderCNN(embed_size)
decoder = DecoderRNN(embed_size, hidden_size, vocab_size)
encoder.to(device)
decoder.to(device)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss().cuda() if torch.cuda.is_available() else nn.CrossEntropyLoss()
params = list(decoder.parameters()) + list(encoder.embed.parameters())
optimizer = torch.optim.Adam(params=params, lr=0.01)

# 训练总步数
total_step = math.ceil(len(data_loader.dataset.caption_lengths) / data_loader.batch_sampler.batch_size)

# 打开日志文件
f = open(log_file, 'w')

# 开始训练
for epoch in range(1, num_epochs + 1):
    for i_step in range(1, total_step + 1):

        # 获取训练索引并设置采样器
        indices = data_loader.dataset.get_train_indices()
        new_sampler = data.sampler.SubsetRandomSampler(indices=indices)
        data_loader.batch_sampler.sampler = new_sampler

        # 获取图像和对应的标注
        images, captions = next(iter(data_loader))
        images = images.to(device)
        captions = captions.to(device)

        # 梯度清零
        decoder.zero_grad()
        encoder.zero_grad()

        # 前向传播
        features = encoder(images)
        outputs = decoder(features, captions)

        # 计算损失
        loss = criterion(outputs.view(-1, vocab_size), captions.view(-1))

        # 反向传播和优化
        loss.backward()
        optimizer.step()

        # 打印和记录训练状态
        stats = 'Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Perplexity: %5.4f' % (
            epoch, num_epochs, i_step, total_step, loss.item(), np.exp(loss.item()))
        print('\r' + stats, end="")
        sys.stdout.flush()
        f.write(stats + '\n')
        f.flush()

        if i_step % print_every == 0:
            print('\r' + stats)

    # 保存模型
    if epoch % save_every == 0:
        torch.save(decoder.state_dict(), os.path.join('E:/nlp/final/CNN-RNN/models', 'decoder-%d.pkl' % epoch))
        torch.save(encoder.state_dict(), os.path.join('E:/nlp/final/CNN-RNN/models', 'encoder-%d.pkl' % epoch))

f.close()
