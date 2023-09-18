import torchvision.transforms as transforms
from torchvision.datasets import CocoCaptions
from torch.utils.data import DataLoader

import preprocess


coco_train_data_path = "C:\\datasets\\coco\\train2017"
coco_val_data_path = "C:\\datasets\\coco\\val2017"
coco_train_annfile_path = "C:\\datasets\\coco\\annotations_trainval2017\\captions_train2017.json"
coco_val_annfile_path = "C:\\datasets\\coco\\annotations_trainval2017\\captions_val2017.json"

batch_size = 1

output_size = (64, 64)
output_chars = " .,:;0#@"


transform = transforms.Compose([
    transforms.Resize(output_size),
    transforms.Grayscale(),
    transforms.ToTensor(),
    preprocess.Ascii(len(output_chars))
])


coco_train = CocoCaptions(
    root=coco_train_data_path,
    annFile=coco_train_annfile_path,
    transform=transform
)

coco_val = CocoCaptions(
    root=coco_val_data_path,
    annFile=coco_val_annfile_path,
    transform=transform
)


dl_train = DataLoader(coco_train, batch_size=batch_size, shuffle=True)
dl_val = DataLoader(coco_val, batch_size=batch_size, shuffle=False)
