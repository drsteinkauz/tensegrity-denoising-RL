import torch

class TensorBuffer:
    def __init__(self, buffer_size=256, device=None):
        self.buffer_size = buffer_size
        self.device = device if device is not None else torch.device("cpu")
        self.buffer = torch.zeros(buffer_size, 3, device=self.device)  # 初始化一个形状为[buffer_size, 3]的张量
        self.current_size = 0  # 当前buffer中存储的有效数据数量

    def add_tensor(self, tensor):
        """
        向buffer中添加一个形状为[1, 3]的tensor
        """
        if tensor.shape != (1, 3):
            raise ValueError("输入的tensor必须是形状为[1, 3]")
        tensor = torch.clamp(tensor,-2,2)
        #print("Tensor:",tensor)
        if self.current_size < self.buffer_size:
            # 如果buffer未满，直接在buffer末尾添加
            self.buffer[self.current_size] = tensor
            self.current_size += 1
        else:
            # 如果buffer已满，覆盖最早的元素
            self.buffer = torch.cat((self.buffer[1:], tensor), dim=0)

    def sample(self):
        """
        返回buffer中所有tensor的平均值，形状为[1, 3]
        """
        if self.current_size == 0:
            return torch.zeros(1, 3, device=self.device)  # 如果buffer为空，返回全零张量
        exp_mean = self.buffer[:self.current_size].mean(dim=0, keepdim=True)
        #print("Mean:",exp_mean)
        return exp_mean

    def to(self, device):
        """
        将buffer中的数据转移到指定的设备（CPU或GPU）
        """
        self.device = device
        self.buffer = self.buffer.to(device)