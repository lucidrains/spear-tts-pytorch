from pathlib import Path

import torch
from torch.utils.data import Dataset

from beartype import beartype

# mock dataset

class MockDataset(Dataset):
    def __init__(self, length: int):
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, ind):
        return torch.randn(1024)

# generated audio-text dataset

class GeneratedAudioTextDataset(Dataset):
    @beartype
    def __init__(
        self,
        folder: str,
        delimiter_id: int = -1
    ):
        self.folder = Path(folder)
        assert self.folder.exists() and self.folder.is_dir()
        self.paths = list(self.folder.glob('*.pt'))
        self.delimiter_id = delimiter_id

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, ind):
        path = self.paths[ind]
        tensor = torch.load(str(path))

        delimiter_mask = tensor == self.delimiter_id
        assert delimiter_mask.any(), f'delimeter (<audio> <delimeter> <text>) not found'

        ind = (delimiter_mask.cumsum(dim = -1) == 0).sum().item()

        return tensor[:ind], tensor[(ind + 1):]
