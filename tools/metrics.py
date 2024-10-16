

import numpy as np 
from torch.nn import functional as F

# 计算混淆矩阵
def fast_hist(a: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
    '''
        a 为标注图
        b 为预测图
        n 为类别
    '''
    # k为掩膜（去除了255这些点，即标签图中的无效点），其中的a >= 0 是为了防止bincount()函数出错
    k = (a >= 0) & (a < n) 
    
    # bincount()函数用于统计数组中每个非负整数的个数，返回统计的结果。
    # 核心代码：将每个像素的位置映射到二维的n×n矩阵，统计每个类别的真实值与预测值的组合。
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)

