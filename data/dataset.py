from torchvision.datasets import CocoCaptions
from torch.utils.data import DataLoader


coco_train_data_path = "C:\\datasets\\coco\\train2017"
coco_val_data_path = "C:\\datasets\\coco\\val2017"
coco_train_annfile_path = "C:\\datasets\\coco\\annotations_trainval2017\\captions_train2017.json"
coco_val_annfile_path = "C:\\datasets\\coco\\annotations_trainval2017\\captions_val2017.json"

batch_size = 32

output_size = (64, 64)
output_chars = " .,:;0#@"


coco_train = CocoCaptions(
    root=coco_train_data_path,
    annFile=coco_train_annfile_path
)

coco_val = CocoCaptions(
    root=coco_val_data_path,
    annFile=coco_val_annfile_path
)


dl_train = DataLoader(coco_train, batch_size=batch_size, shuffle=True)
dl_val = DataLoader(coco_val, batch_size=batch_size, shuffle=False)
