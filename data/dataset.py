import os
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

class IntelImageClassificationDataset:
    def __init__(self, root_dir, batch_size=64, num_workers=4, transform=None):
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform if transform else self.default_transform()

        self.train_dataset = ImageFolder(root=os.path.join(root_dir, "seg_train", "seg_train"), transform=self.transform)
        self.test_dataset = ImageFolder(root=os.path.join(root_dir, "seg_test", "seg_test"), transform=self.transform)

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def default_transform(self):
        return transforms.Compose([
            transforms.Resize((150, 150)),
            transforms.ToTensor(),
        ])

    def get_train_loader(self):
        return self.train_loader

    def get_test_loader(self):
        return self.test_loader

    def get_classes(self):
        return self.train_dataset.classes

    def __len__(self):
        return len(self.train_dataset) + len(self.test_dataset)
    

import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class IntelImageDataset(Dataset):
    """Dataset for Intel Image Classification"""
    
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir: Directory with all the images organized in class folders
            transform: Optional transform to be applied on a sample
        """
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            for img_name in os.listdir(class_dir):
                if img_name.endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append((
                        os.path.join(class_dir, img_name),
                        self.class_to_idx[class_name]
                    ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label