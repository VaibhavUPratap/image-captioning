import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import os
from tqdm import tqdm

from dataset.flickr8k_handler import Flickr8kDataset
from tokenizer.tokenizer import Tokenizer
from models.encoder.encoder import EncoderCNN
from models.decoder.decoder import DecoderRNN
from models.fusion.model import ImageCaptioner

def train():
    # 1. Hyperparameters
    embed_size = 256
    hidden_size = 512
    num_layers = 1
    batch_size = 32
    learning_rate = 3e-4
    num_epochs = 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Paths
    root_dir = "dataset/raw/flickr8k/Images"
    captions_file = "dataset/raw/flickr8k/Flickr8k.token.txt"
    checkpoint_path = "checkpoints/best/model.pth"

    # 3. Data Transformations
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # 4. Initialize Dataset and Tokenizer
    dataset = Flickr8kDataset(root_dir, captions_file, transform=transform)
    
    # Build Vocabulary
    tokenizer = Tokenizer()
    tokenizer.build_vocab([cap for _, cap in dataset.data])
    vocab_size = len(tokenizer.vocab)

    # 5. Collate function for padding
    def collate_fn(data):
        images, captions = zip(*data)
        images = torch.stack(images, 0)
        
        # Tokenize and pad captions
        tokenized_captions = [torch.tensor(tokenizer.encode(cap)) for cap in captions]
        padded_captions = torch.nn.utils.rnn.pad_sequence(tokenized_captions, batch_first=True, padding_value=tokenizer.pad_idx)
        
        return images, padded_captions

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # 6. Initialize Model
    encoder = EncoderCNN(embed_size).to(device)
    decoder = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers).to(device)
    model = ImageCaptioner(encoder, decoder).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_idx)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 7. Training Loop
    model.train()
    for epoch in range(num_epochs):
        loop = tqdm(loader, total=len(loader))
        for images, captions in loop:
            images, captions = images.to(device), captions.to(device)
            
            # Forward pass
            outputs = model(images, captions)
            
            # Loss Calculation (flatten outputs and captions)
            # outputs: [batch, seq_len, vocab_size] -> [batch * seq_len, vocab_size]
            # captions: [batch, seq_len] -> [batch * seq_len]
            loss = criterion(outputs.view(-1, vocab_size), captions.view(-1))
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loop.set_description(f"Epoch [{epoch+1}/{num_epochs}]")
            loop.set_postfix(loss=loss.item())

        # Save Checkpoint
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Epoch {epoch+1} saved.")

if __name__ == "__main__":
    train()
