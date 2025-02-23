import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, TensorDataset
from attention_pytorch import Attention  # 确保这是您保存 attention_pytorch.py 的位置
from datasets import load_dataset
import math  # 添加math模块导入

# 超参数设置
max_features = 20000
maxlen = 80
batch_size = 32
embedding_dim = 128
num_heads = 8
size_per_head = 16
dropout_rate = 0.5
epochs = 10
learning_rate = 0.001

# 1. 数据加载和预处理
print('Loading data...')
# 使用 Hugging Face datasets 加载 IMDB 数据集
dataset = load_dataset("imdb")

# 定义 tokenize 函数
def tokenize(batch):
    return {'text': [text.split()[:maxlen] for text in batch['text']]}

# 应用 tokenize 函数
tokenized_datasets = dataset.map(tokenize, batched=True)

# 定义 encode 函数
def encode(batch, word_to_index):
    return {'encoded': [[word_to_index.get(word, 1) for word in text] for text in batch['text']]}

# 构建词汇表
def build_vocab(tokenized_datasets, max_features):
    word_counts = {}
    for example in tokenized_datasets["train"]["text"]:
        for word in example:
            if word in word_counts:
                word_counts[word] += 1
            else:
                word_counts[word] = 1

    sorted_words = sorted(word_counts, key=word_counts.get, reverse=True)
    top_words = sorted_words[:max_features-2]  # 留出两个位置给 padding 和 unknown
    word_to_index = {word: i+2 for i, word in enumerate(top_words)}
    word_to_index["<pad>"] = 0
    word_to_index["<unk>"] = 1
    return word_to_index

word_to_index = build_vocab(tokenized_datasets, max_features)

# 应用 encode 函数
encoded_datasets = tokenized_datasets.map(encode, batched=True, fn_kwargs={'word_to_index': word_to_index})

# 定义 pad 函数
def pad(batch, maxlen):
    padded_sequences = [seq + [0]*(maxlen-len(seq)) if len(seq) < maxlen else seq[:maxlen] for seq in batch['encoded']]
    return {'padded': padded_sequences}

# 应用 pad 函数
padded_datasets = encoded_datasets.map(pad, batched=True, fn_kwargs={'maxlen': maxlen})

# 转换为 PyTorch tensors
x_train = torch.tensor(padded_datasets["train"]["padded"], dtype=torch.long)
y_train = torch.tensor(dataset["train"]["label"], dtype=torch.float)
x_test = torch.tensor(padded_datasets["test"]["padded"], dtype=torch.long)
y_test = torch.tensor(dataset["test"]["label"], dtype=torch.float)

print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

# 创建数据加载器
train_data = TensorDataset(x_train, y_train)
test_data = TensorDataset(x_test, y_test)
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(test_data, batch_size=batch_size)

# 添加位置编码实现
class SinCosPositionEmbedding(nn.Module):
    def __init__(self, v_dim):
        super(SinCosPositionEmbedding, self).__init__()
        self.v_dim = v_dim

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        position_ids = torch.arange(seq_len, dtype=torch.float32, device=x.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_len)
        
        # 计算位置编码
        div_term = torch.exp(torch.arange(0, self.v_dim, 2, dtype=torch.float32, device=x.device) * 
                           -(math.log(10000.0) / self.v_dim))
        
        pe = torch.zeros(batch_size, seq_len, self.v_dim, device=x.device)
        pe[:, :, 0::2] = torch.sin(position_ids.unsqueeze(-1) * div_term)
        pe[:, :, 1::2] = torch.cos(position_ids.unsqueeze(-1) * div_term)
        
        return x + pe

# 2. 定义模型
class AttentionModel(nn.Module):
    def __init__(self, max_features, embedding_dim, num_heads, size_per_head, dropout_rate, use_position_embedding=True):
        super(AttentionModel, self).__init__()
        self.embedding = nn.Embedding(max_features, embedding_dim)
        self.use_position_embedding = use_position_embedding
        if use_position_embedding:
            self.position_embedding = SinCosPositionEmbedding(embedding_dim)
        self.attention = Attention(num_heads, size_per_head)
        self.global_pooling = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(num_heads * size_per_head, 1)

    def forward(self, x):
        embeddings = self.embedding(x)
        # 根据配置决定是否添加位置编码
        if self.use_position_embedding:
            embeddings = self.position_embedding(embeddings)
        # Attention layer expects q, k, v as input
        O_seq = self.attention([embeddings, embeddings, embeddings])
        O_seq = O_seq.permute(0, 2, 1)  # Change to (batch_size, embedding_dim, seq_len)
        O_seq = self.global_pooling(O_seq).squeeze(-1)
        O_seq = self.dropout(O_seq)
        output = torch.sigmoid(self.linear(O_seq))
        return output

# 训练和评估函数
def train_and_evaluate(model, train_loader, test_loader, optimizer, criterion, epochs, device, use_position_embedding):
    print(f'{"Training with Position Embedding" if use_position_embedding else "Training without Position Embedding"}...')
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data).squeeze()
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader)}")

        # 评估模型
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data).squeeze()
                predicted = (output > 0.5).float()
                total += target.size(0)
                correct += (predicted == target).sum().item()

        print('Accuracy of the model on the test sequences: {} %'.format(100 * correct / total))

# 3. 模型初始化和训练（有位置编码）
use_position_embedding = True
model_with_pe = AttentionModel(max_features, embedding_dim, num_heads, size_per_head, dropout_rate, use_position_embedding)
optimizer_with_pe = optim.Adam(model_with_pe.parameters(), lr=learning_rate)
criterion = nn.BCELoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_with_pe.to(device)

train_and_evaluate(model_with_pe, train_loader, test_loader, optimizer_with_pe, criterion, epochs, device, use_position_embedding)

# 4. 模型初始化和训练（没有位置编码）
use_position_embedding = False
model_without_pe = AttentionModel(max_features, embedding_dim, num_heads, size_per_head, dropout_rate, use_position_embedding)
optimizer_without_pe = optim.Adam(model_without_pe.parameters(), lr=learning_rate)
model_without_pe.to(device)

train_and_evaluate(model_without_pe, train_loader, test_loader, optimizer_without_pe, criterion, epochs, device, use_position_embedding)