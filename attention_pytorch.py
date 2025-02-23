import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def to_mask(x, mask, mode='mul'):
    """通用mask函数
    mask.shape=[batch_size, seq_len]或[batch_size, seq_len, 1]
    """
    if mask is None:
        return x
    else:
        for _ in range(x.dim() - mask.dim()):
            mask = mask.unsqueeze(mask.dim())
        if mode == 'mul':
            return x * mask
        else:
            return x - (1 - mask) * 1e10

def extract_seq_patches(x, kernel_size, rate):
    """x.shape = [batch_size, seq_len, seq_dim]
    滑动地把每个窗口的x取出来，为做局部attention作准备。
    """
    batch_size, seq_len, seq_dim = x.size()
    k_size = kernel_size + (rate - 1) * (kernel_size - 1)
    p_right = (k_size - 1) // 2
    p_left = k_size - 1 - p_right
    
    # 填充序列
    x = F.pad(x, (0, 0, p_left, p_right), mode='constant', value=0)
    
    # 提取patches
    patches = []
    for i in range(0, k_size, rate):
        patch = x[:, i:i + seq_len]
        patches.append(patch)
    
    x = torch.cat(patches, dim=2)
    return x.view(batch_size, seq_len, kernel_size, seq_dim)

class Attention(nn.Module):
    """多头注意力机制
    """
    def __init__(self, heads, size_per_head, key_size=None, mask_right=False):
        super(Attention, self).__init__()
        self.heads = heads
        self.size_per_head = size_per_head
        self.out_dim = heads * size_per_head
        self.key_size = key_size if key_size else size_per_head
        self.mask_right = mask_right
        
        self.q_dense = nn.Linear(self.out_dim, self.key_size * self.heads, bias=False)
        self.k_dense = nn.Linear(self.out_dim, self.key_size * self.heads, bias=False)
        self.v_dense = nn.Linear(self.out_dim, self.out_dim, bias=False)

    def forward(self, inputs):
        q, k, v = inputs[:3]
        v_mask, q_mask = None, None
        if len(inputs) > 3:
            v_mask = inputs[3]
            if len(inputs) > 4:
                q_mask = inputs[4]

        # 线性变换
        qw = self.q_dense(q)
        kw = self.k_dense(k)
        vw = self.v_dense(v)
        
        # 形状变换
        qw = qw.view(qw.size(0), qw.size(1), self.heads, self.key_size)
        kw = kw.view(kw.size(0), kw.size(1), self.heads, self.key_size)
        vw = vw.view(vw.size(0), vw.size(1), self.heads, self.size_per_head)
        
        # 维度置换
        qw = qw.permute(0, 2, 1, 3)
        kw = kw.permute(0, 2, 1, 3)
        vw = vw.permute(0, 2, 1, 3)
        
        # Attention
        a = torch.matmul(qw, kw.transpose(-1, -2)) / self.key_size**0.5
        
        if v_mask is not None:
            a = to_mask(a.permute(0, 3, 2, 1), v_mask, 'add')
            a = a.permute(0, 3, 2, 1)
            
        if self.mask_right is not False:
            if self.mask_right is True:
                mask = torch.ones_like(a[:1, :1])
                mask = (1 - torch.triu(mask, diagonal=1)) * 1e10
                a = a - mask
            else:
                mask = (1 - torch.tensor(self.mask_right, device=a.device)) * 1e10
                mask = mask.unsqueeze(0).unsqueeze(0)
                a = a - mask

        a = F.softmax(a, dim=-1)
        
        # 完成输出
        o = torch.matmul(a, vw)
        o = o.permute(0, 2, 1, 3)
        o = o.reshape(o.size(0), o.size(1), self.out_dim)
        
        if q_mask is not None:
            o = to_mask(o, q_mask, 'mul')
            
        return o

class SelfAttention(nn.Module):
    """多头自注意力机制
    """
    def __init__(self, heads, size_per_head, key_size=None, mask_right=False):
        super(SelfAttention, self).__init__()
        self.heads = heads
        self.size_per_head = size_per_head
        self.out_dim = heads * size_per_head
        self.key_size = key_size if key_size else size_per_head
        self.mask_right = mask_right
        self.attention = Attention(heads, size_per_head, key_size, mask_right)

    def forward(self, inputs):
        if isinstance(inputs, (list, tuple)):
            x, x_mask = inputs
            o = self.attention([x, x, x, x_mask, x_mask])
        else:
            x = inputs
            o = self.attention([x, x, x])
        return o

class AtrousSelfAttention(nn.Module):
    """空洞多头自注意力机制
    说明：每个元素只跟相对距离为rate的倍数的元素有关联。
    """
    def __init__(self, heads, size_per_head, rate=1, key_size=None, mask_right=False):
        super(AtrousSelfAttention, self).__init__()
        self.heads = heads
        self.size_per_head = size_per_head
        self.out_dim = heads * size_per_head
        self.key_size = key_size if key_size else size_per_head
        self.rate = rate
        self.mask_right = mask_right
        self.attention = Attention(heads, size_per_head, key_size, mask_right)

    def forward(self, inputs):
        if isinstance(inputs, (list, tuple)):
            x, x_mask = inputs
        else:
            x, x_mask = inputs, None
        
        batch_size, seq_len, seq_dim = x.size()
        
        # 补足长度，保证可以reshape
        pad_len = self.rate - seq_len % self.rate
        x = F.pad(x, (0, 0, 0, pad_len), mode='constant', value=0)
        if x_mask is not None:
            x_mask = F.pad(x_mask, (0, pad_len), mode='constant', value=0)
        
        new_seq_len = x.size(1)
        
        # 变换shape
        x = x.view(batch_size, new_seq_len // self.rate, self.rate, seq_dim)
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(batch_size * self.rate, new_seq_len // self.rate, seq_dim)
        
        if x_mask is not None:
            x_mask = x_mask.view(batch_size, new_seq_len // self.rate, self.rate, 1)
            x_mask = x_mask.permute(0, 2, 1, 3)
            x_mask = x_mask.reshape(batch_size * self.rate, new_seq_len // self.rate, 1)
        
        # 做attention
        if x_mask is not None:
            x = self.attention([x, x, x, x_mask, x_mask])
        else:
            x = self.attention([x, x, x])
        
        # 恢复shape
        x = x.view(batch_size, self.rate, new_seq_len // self.rate, self.out_dim)
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(batch_size, new_seq_len, self.out_dim)
        
        x = x[:, :seq_len]
        return x

class LocalSelfAttention(nn.Module):
    """局部多头自注意力机制
    说明：每个元素只跟相对距离不超过neighbors的元素有关联，这里的rate
    是真正的膨胀率（跟膨胀卷积一样），如果不了解可以忽略，默认为1就好。
    """
    def __init__(self, heads, size_per_head, neighbors=1, rate=1, key_size=None, mask_right=False):
        super(LocalSelfAttention, self).__init__()
        self.heads = heads
        self.size_per_head = size_per_head
        self.out_dim = heads * size_per_head
        self.key_size = key_size if key_size else size_per_head
        self.neighbors = neighbors
        self.rate = rate
        self.mask_right = mask_right
        
        if self.mask_right:
            mask_right_np = np.ones((1, 1 + 2 * self.neighbors))
            mask_right_np[:, - self.neighbors:] = 0
            self.mask_right_tensor = torch.tensor(mask_right_np, dtype=torch.float32)
        else:
            self.mask_right_tensor = None
        
        self.attention = Attention(heads, size_per_head, key_size, self.mask_right_tensor)

    def forward(self, inputs):
        if isinstance(inputs, (list, tuple)):
            x, x_mask = inputs
        else:
            x, x_mask = inputs, None
        
        # 提取局部特征
        kernel_size = 1 + 2 * self.neighbors
        xp = extract_seq_patches(x, kernel_size, self.rate)
        
        if x_mask is not None:
            xp_mask = extract_seq_patches(x_mask, kernel_size, self.rate)
        
        # 变换shape
        batch_size, seq_len, seq_dim = x.size()
        x = x.view(batch_size, seq_len, 1, seq_dim)
        xp = xp.view(batch_size, seq_len, kernel_size, seq_dim)
        
        if x_mask is not None:
            x_mask = x_mask.view(batch_size, seq_len, 1, 1)
            xp_mask = xp_mask.view(batch_size, seq_len, kernel_size, 1)
        
        # 做attention
        if x_mask is not None:
            x = self.attention([x, xp, xp, xp_mask])
        else:
            x = self.attention([x, xp, xp])
        
        # 恢复shape
        x = x.view(batch_size, seq_len, self.out_dim)
        
        if x_mask is not None:
            x = to_mask(x, x_mask.squeeze(-1), 'mul')
        
        return x

class SparseSelfAttention(nn.Module):
    """稀疏多头自注意力机制
    来自文章《Generating Long Sequences with Sparse Transformers》
    说明：每个元素只跟相对距离为rate的倍数的元素、以及相对距离不超过rate的元素有关联。
    """
    def __init__(self, heads, size_per_head, rate=2, key_size=None, mask_right=False):
        super(SparseSelfAttention, self).__init__()
        self.heads = heads
        self.size_per_head = size_per_head
        self.out_dim = heads * size_per_head
        self.key_size = key_size if key_size else size_per_head
        assert rate != 1, "if rate=1, please use SelfAttention directly"
        self.rate = rate
        self.neighbors = rate - 1
        self.mask_right = mask_right
        
        self.q_dense = nn.Linear(self.out_dim, self.key_size * self.heads, bias=False)
        self.k_dense = nn.Linear(self.out_dim, self.key_size * self.heads, bias=False)
        self.v_dense = nn.Linear(self.out_dim, self.out_dim, bias=False)

    def forward(self, inputs):
        if isinstance(inputs, (list, tuple)):
            x, x_mask = inputs
        else:
            x, x_mask = inputs, None
        
        batch_size, seq_len, seq_dim = x.size()
        
        # 补足长度，保证可以reshape
        pad_len = self.rate - seq_len % self.rate
        x = F.pad(x, (0, 0, 0, pad_len), mode='constant', value=0)
        if x_mask is not None:
            x_mask = F.pad(x_mask, (0, pad_len), mode='constant', value=0)
        
        new_seq_len = x.size(1)
        
        # 线性变换
        qw = self.q_dense(x)
        kw = self.k_dense(x)
        vw = self.v_dense(x)
        
        # 提取局部特征
        kernel_size = 1 + 2 * self.neighbors
        kwp = extract_seq_patches(kw, kernel_size, self.rate)
        vwp = extract_seq_patches(vw, kernel_size, self.rate)
        
        if x_mask is not None:
            xp_mask = extract_seq_patches(x_mask, kernel_size, self.rate)
        
        # 形状变换
        qw = qw.view(batch_size, new_seq_len // self.rate, self.rate, self.heads, self.key_size)
        kw = kw.view(batch_size, new_seq_len // self.rate, self.rate, self.heads, self.key_size)
        vw = vw.view(batch_size, new_seq_len // self.rate, self.rate, self.heads, self.size_per_head)
        kwp = kwp.view(batch_size, new_seq_len // self.rate, self.rate, kernel_size, self.heads, self.key_size)
        vwp = vwp.view(batch_size, new_seq_len // self.rate, self.rate, kernel_size, self.heads, self.size_per_head)
        
        if x_mask is not None:
            x_mask = x_mask.view(batch_size, new_seq_len // self.rate, self.rate, 1, 1)
            xp_mask = xp_mask.view(batch_size, new_seq_len // self.rate, self.rate, kernel_size, 1, 1)
        
        # 维度置换
        qw = qw.permute(0, 3, 2, 1, 4)
        kw = kw.permute(0, 3, 2, 1, 4)
        vw = vw.permute(0, 3, 2, 1, 4)
        qwp = qw.unsqueeze(4)
        kwp = kwp.permute(0, 4, 2, 1, 3, 5)
        vwp = vwp.permute(0, 4, 2, 1, 3, 5)
        
        if x_mask is not None:
            x_mask = x_mask.permute(0, 3, 2, 1, 4)
            xp_mask = xp_mask.permute(0, 4, 2, 1, 3, 5)
        
        # Attention1
        a = torch.matmul(qw, kw.transpose(-1, -2)) / self.key_size**0.5
        
        if x_mask is not None:
            a = to_mask(a.permute(0, 1, 2, 4, 3), x_mask, 'add')
            a = a.permute(0, 1, 2, 4, 3)
        
        if self.mask_right:
            mask = torch.ones_like(a[:1, :1, :1])
            mask = (1 - torch.triu(mask, diagonal=1)) * 1e10
            a = a - mask
        
        # Attention2
        ap = torch.matmul(qwp, kwp.transpose(-1, -2)) / self.key_size**0.5
        
        if x_mask is not None:
            ap = to_mask(ap.permute(0, 1, 2, 3, 5, 4), xp_mask, 'add')
            ap = ap.permute(0, 1, 2, 3, 5, 4)
        
        if self.mask_right:
            mask = np.ones((1, kernel_size))
            mask[:, - self.neighbors:] = 0
            mask = (1 - torch.tensor(mask, dtype=torch.float32, device=x.device)) * 1e10
            for _ in range(4):
                mask = mask.unsqueeze(0)
            ap = ap - mask
        
        ap = ap[..., 0, :]
        
        # 合并两个Attention
        A = torch.cat([a, ap], dim=-1)
        A = F.softmax(A, dim=-1)
        a, ap = A[..., :a.size(-1)], A[..., a.size(-1):]
        
        # 完成输出1
        o1 = torch.matmul(a, vw)
        
        # 完成输出2
        ap = ap.unsqueeze(-2)
        o2 = torch.matmul(ap, vwp)
        o2 = o2[..., 0, :]
        
        # 完成输出
        o = o1 + o2
        
        if x_mask is not None:
            o = to_mask(o, x_mask.squeeze(-1), 'mul')
        
        o = o.permute(0, 3, 2, 1, 4)
        o = o.reshape(batch_size, new_seq_len, self.out_dim)
        o = o[:, :seq_len]
        return o 