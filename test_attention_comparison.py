import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from attention_pytorch import (
    SelfAttention,
    AtrousSelfAttention,
    LocalSelfAttention,
    SparseSelfAttention
)
import math

class SimpleDataset(Dataset):
    def __init__(self, seq_len=50, num_samples=1000):
        self.data = torch.randn(num_samples, seq_len, 32)  # 随机生成序列数据
        # 生成简单的二分类标签
        self.labels = torch.tensor([1 if x.mean() > 0 else 0 for x in self.data])
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class SinCosPositionEmbedding(nn.Module):
    def __init__(self, v_dim, seq_len):
        super(SinCosPositionEmbedding, self).__init__()
        self.v_dim = v_dim
        self.seq_len = seq_len
        self.register_buffer('pe', self.create_positional_encoding())

    def create_positional_encoding(self):
        position = torch.arange(0, self.seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.v_dim, 2).float() * (-math.log(10000.0) / self.v_dim))
        pe = torch.zeros(self.seq_len, self.v_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:x.size(1), :].unsqueeze(0)
        return x

class AttentionModel(nn.Module):
    def __init__(self, attention_type, seq_len=50, dim=32, use_pos_embedding=True):
        super().__init__()
        self.use_pos_embedding = use_pos_embedding
        if use_pos_embedding:
            self.pos_embedding = SinCosPositionEmbedding(dim, seq_len)  # 使用SinCosPositionEmbedding
        
        if attention_type == "self":
            self.attention = SelfAttention(heads=4, size_per_head=32)
        elif attention_type == "atrous":
            self.attention = AtrousSelfAttention(heads=4, size_per_head=32, rate=2)
        elif attention_type == "local":
            self.attention = LocalSelfAttention(heads=4, size_per_head=32, neighbors=2)
        elif attention_type == "sparse":
            self.attention = SparseSelfAttention(heads=4, size_per_head=32, rate=2)
        
        self.fc = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()
        self.input_fc = nn.Linear(dim, 4 * 32)
        self.seq_len = seq_len # 记录序列长度
        
    def forward(self, x):
        if self.use_pos_embedding:
            x = self.pos_embedding(x) # 使用位置编码
        
        x = self.input_fc(x)
        x = self.attention(x)
        x = x.mean(dim=1)  # 对序列取平均
        x = self.fc(x)
        return self.sigmoid(x)

def train_and_evaluate(model, train_loader, test_loader, epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    best_acc = 0
    history = {'train_loss': [], 'test_acc': []}
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.float().to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        history['train_loss'].append(avg_loss)
        
        # 评估
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x).squeeze()
                predicted = (outputs > 0.5).float()
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        
        accuracy = 100 * correct / total
        history['test_acc'].append(accuracy)
        best_acc = max(best_acc, accuracy)
        
        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'Average Loss: {avg_loss:.4f}')
        print(f'Test Accuracy: {accuracy:.2f}%')
        print('------------------------')
    
    return best_acc, history

def main():
    # 数据准备
    train_dataset = SimpleDataset(num_samples=1000)
    test_dataset = SimpleDataset(num_samples=200)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    attention_types = ["self", "atrous", "sparse"]
    results = {}
    
    for att_type in attention_types:
        print(f"\n测试 {att_type} Attention:")
        
        # 有Position Embedding的测试
        print("\n使用Position Embedding:")
        model_with_pos = AttentionModel(att_type, use_pos_embedding=True)
        best_acc_with_pos, history_with_pos = train_and_evaluate(
            model_with_pos, train_loader, test_loader
        )
        
        # 无Position Embedding的测试
        print("\n不使用Position Embedding:")
        model_without_pos = AttentionModel(att_type, use_pos_embedding=False)
        best_acc_without_pos, history_without_pos = train_and_evaluate(
            model_without_pos, train_loader, test_loader
        )
        
        results[att_type] = {
            'with_pos': best_acc_with_pos,
            'without_pos': best_acc_without_pos
        }
    
    # 打印最终结果比较
    print("\n最终结果比较:")
    print("=" * 50)
    print("Attention类型  | 有Position Embedding | 无Position Embedding")
    print("-" * 50)
    for att_type, scores in results.items():
        print(f"{att_type:12} | {scores['with_pos']:17.2f}% | {scores['without_pos']:18.2f}%")
    print("=" * 50)

if __name__ == "__main__":
    main() 