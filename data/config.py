# coco data directories
coco_train_data_path = "C:\\datasets\\coco\\train2017"
coco_val_data_path = "C:\\datasets\\coco\\val2017"
coco_train_annfile_path = "C:\\datasets\\coco\\annotations_trainval2017\\captions_train2017.json"
coco_val_annfile_path = "C:\\datasets\\coco\\annotations_trainval2017\\captions_val2017.json"

# train/test batch size
batch_size = 32

# output image
output_size = (64, 64)
output_chars = " .,:;0#@"

# spacy nlp model name
spacy_model = "en_core_web_sm"

# max keywords to extract from coco captions
n_keywords = 4
