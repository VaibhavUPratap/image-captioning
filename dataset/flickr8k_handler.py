import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class Flickr8kDataset(Dataset):
    """
    Custom Dataset class for Flickr8k.
    Structure:
    - images/ folder with JPG files
    - Flickr8k.token.txt with format: image.jpg#0   caption
    """
    def __init__(self, root_dir, captions_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        raw_data = []
        with open(captions_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 2:
                    continue
                img_id = parts[0].split('#')[0]
                caption = parts[1]
                raw_data.append((img_id, caption))
        
        # Filter out missing images to avoid runtime errors
        print(f"Validating dataset images in {root_dir}...")
        self.data = []
        missing_count = 0
        existing_files = set(os.listdir(root_dir))
        
        for img_id, caption in raw_data:
            # Normalize filename (same logic as in __getitem__)
            clean_id = img_id
            if not clean_id.lower().endswith('.jpg'):
                if '.jpg' in clean_id.lower():
                    clean_id = clean_id[:clean_id.lower().find('.jpg') + 4]
            
            if clean_id in existing_files:
                self.data.append((clean_id, caption))
            else:
                missing_count += 1
        
        if missing_count > 0:
            print(f"Skipped {missing_count} entries because image files were missing.")
        print(f"Final dataset size: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_id, caption = self.data[idx]
        img_path = os.path.join(self.root_dir, img_id)
        
        image = Image.open(img_path).convert("RGB")
        
        if self.transform is not None:
            image = self.transform(image)
        
        return image, caption

def get_flickr8k_loader(root_dir, captions_file, transform, batch_size=32, shuffle=True):
    dataset = Flickr8kDataset(root_dir, captions_file, transform)
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )
    return loader, dataset

if __name__ == "__main__":
    # Define basic transform for testing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    print("Flickr8k Handler defined. Ready to use once dataset is downloaded.")
