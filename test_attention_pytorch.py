import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, TensorDataset
from attention_pytorch import Attention  # 确保这是您保存 attention_pytorch.py 的位置
from datasets import load_dataset

# 超参数设置
max_features = 20000
maxlen = 80
batch_size = 32
embedding_dim = 128
num_heads = 8
size_per_head = 16
dropout_rate = 0.5
epochs = 5
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

# 2. 定义模型
class AttentionModel(nn.Module):
    def __init__(self, max_features, embedding_dim, num_heads, size_per_head, dropout_rate):
        super(AttentionModel, self).__init__()
        self.embedding = nn.Embedding(max_features, embedding_dim)
        self.attention = Attention(num_heads, size_per_head)
        self.global_pooling = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(num_heads * size_per_head, 1)

    def forward(self, x):
        embeddings = self.embedding(x)
        # Attention layer expects q, k, v as input
        O_seq = self.attention([embeddings, embeddings, embeddings])
        O_seq = O_seq.permute(0, 2, 1)  # Change to (batch_size, embedding_dim, seq_len)
        O_seq = self.global_pooling(O_seq).squeeze(-1)
        O_seq = self.dropout(O_seq)
        output = torch.sigmoid(self.linear(O_seq))
        return output

# 3. 模型初始化
model = AttentionModel(max_features, embedding_dim, num_heads, size_per_head, dropout_rate)

# 4. 定义优化器和损失函数
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.BCELoss()

# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 5. 训练循环
print('Train...')
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

    # 6. 评估模型
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