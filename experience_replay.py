from collections import deque
import random

class ReplayMemory():
    def __init__(self, maxlen, seed=None):
        self.memory = deque([], maxlen=maxlen)
        
        # 随机初始化
        if seed is not None:
            random.seed(seed)
    
    def append(self, transition):
        # 一个 experience
        self.memory.append(transition)
    
    def sample(self, sample_size):
        # 随机采样
        return random.sample(self.memory, sample_size)
    
    def __len__(self):
        return len(self.memory)