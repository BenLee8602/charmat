import torch


class Ascii:
    def __init__(self, char_count: int):
        self.char_count = char_count
    
    def __call__(self, x):
        return torch.round(x * self.char_count)
