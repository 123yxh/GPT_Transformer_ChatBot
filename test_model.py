import torch
from model import GPTModel

"""
输出打印模型参数--验证
"""
def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 模型参数
    model_param = {
        "d_model": 768,  # 嵌入层大小
        "d_ff": 2048,  # 前馈神经网络大小
        "d_k": 64,  # K 的大小
        "d_v": 64,  # V 的大小
        "n_layers": 6,  # 解码层的数量
        "n_heads": 8,  # 多头注意力的头数
        "max_pos": 1800,  # 位置编码的长度
        "device": device,  # 设备
        "vocab_size": 4825  # 词表大小
    }
    model = GPTModel(**model_param)
    total_params = sum(p.numel() for p in model.parameters())
    print(model)
    print("total_params: ", total_params)


if __name__ == '__main__':
    main()
