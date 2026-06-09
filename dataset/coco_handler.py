import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO

class COCODataset(Dataset):
    """
    Custom Dataset class for MS COCO 2017.
    """
    def __init__(self, root_dir, ann_file, transform=None):
        self.root_dir = root_dir
        self.coco = COCO(ann_file)
        self.ids = list(self.coco.imgs.keys())
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        
        # COCO has multiple captions per image, we'll pick one
        caption = anns[0]['caption']
        
        path = self.coco.loadImgs(img_id)[0]['file_name']
        image = Image.open(os.path.join(self.root_dir, path)).convert("RGB")
        
        if self.transform is not None:
            image = self.transform(image)
        
        return image, caption

def get_coco_loader(root_dir, ann_file, transform, batch_size=32, shuffle=True):
    dataset = COCODataset(root_dir, ann_file, transform)
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )
    return loader, dataset

if __name__ == "__main__":
    print("COCO Handler defined. Note: Requires 'pycocotools' for operation.")
