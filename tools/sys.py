import os
from datetime import datetime

def create_folder_with_current_time(base_path: str) -> str:
    # 获取当前时间并格式化
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    # 创建完整的文件夹路径
    folder_path = os.path.join(base_path, current_time)
    
    # 创建文件夹
    os.makedirs(folder_path, exist_ok=True)
    
    return folder_path