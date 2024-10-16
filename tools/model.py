import torch

from torchsummary import summary
from thop import profile



def model_info(model, input_size):
    """
    展示模型的参数量、计算复杂度(FLOPs)、模型结构等信息
    
    Args:
    - model (torch.nn.Module): 需要展示信息的 PyTorch 模型
    - input_size (tuple): 输入张量的尺寸, 例如 (3, 224, 224)
    
    Returns:
    - None
    """
    # 1. 展示模型结构与参数量
    print("Model Summary:")
    summary(model, input_size=input_size, device="cpu")
    
    # 2. 计算 FLOPs 和 参数数量
    dummy_input = torch.randn(1, *input_size)  # 创建虚拟输入数据
    flops, params = profile(model, inputs=(dummy_input,))
    
    # 转换 FLOPs 和 参数数量为更加可读的格式
    print(f"\nTotal Parameters: {params:,}")
    print(f"Total FLOPs: {flops / 1e9:.3f} GFLOPs")  # 转换为 Giga FLOPs
    