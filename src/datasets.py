import os
from torch.utils.data import Dataset
from PIL import Image

class FolderDataset(Dataset):
    # root_dir: e.g., data/offense or data/defense
    def __init__(self, root_dir, transform=None, class_map=None):
        self.root = root_dir
        self.transform = transform
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        if class_map:
            self.classes = [c for c in self.classes if c in class_map]
        self.index = []
        for ci, c in enumerate(self.classes):
            cdir = os.path.join(root_dir, c)
            for fname in os.listdir(cdir):
                if fname.lower().endswith(('.png','.jpg','.jpeg','.bmp','.webp')):
                    self.index.append((os.path.join(cdir, fname), ci))
    def __len__(self):
        return len(self.index)
    def __getitem__(self, i):
        path, y = self.index[i]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, y
