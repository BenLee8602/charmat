import torchvision.transforms as transforms
from torchvision.datasets import CocoCaptions
from torch.utils.data import DataLoader

import config as cfg
import preprocess


transform = transforms.Compose([
    transforms.Resize(cfg.output_size),
    transforms.Grayscale(),
    transforms.ToTensor(),
    preprocess.ascii_indexing
])

target_transform = transforms.Compose([
    preprocess.flatten_captions,
    preprocess.extract_keywords,
    preprocess.keyword_vector
])


coco_train = CocoCaptions(
    root=cfg.coco_train_data_path,
    annFile=cfg.coco_train_annfile_path,
    transform=transform,
    target_transform=target_transform
)

coco_val = CocoCaptions(
    root=cfg.coco_val_data_path,
    annFile=cfg.coco_val_annfile_path,
    transform=transform,
    target_transform=target_transform
)


dl_train = DataLoader(coco_train, batch_size=cfg.batch_size, shuffle=True)
dl_val = DataLoader(coco_val, batch_size=cfg.batch_size, shuffle=False)
