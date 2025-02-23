# attention

some attention implements

《Attention is All You Need》中的Attention机制的实现

http://kexue.fm/archives/4765/

## 测试环境

python 2.7 + tensorflow 1.8+ + keras 2.2.4

注：本项目不再跟进新版本的tensorflow/keras，如果有新版本的需求，可以到<a href="https://github.com/bojone/bert4keras/blob/master/bert4keras/layers.py">bert4keras的layers.py</a>处复制对应的函数。

## 交流
QQ交流群：67729435，微信群请加机器人微信号spaces_ac_cn

## Conda 环境配置

要运行此项目，建议使用conda根据`environment.yaml`文件创建一个新的虚拟环境。可以使用以下命令创建并激活环境：

```bash
conda env create -f environment.yaml
conda activate attention_env
```

确保`environment.yaml`文件中包含所需的所有依赖项。

## 使用说明（PyTorch版本）

本项目的PyTorch版本实现了《Attention is All You Need》中的多头注意力机制（Multi-Head Attention），并支持局部注意力的计算。

要运行PyTorch版本的代码，请确保已安装上述环境，并使用以下命令：

```bash
python test_attention_pytorch.py
```

请根据需要修改代码路径和参数设置。具体实现细节可以在`attention_pytorch.py`文件中查看。
