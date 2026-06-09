import torch
from torchvision import transforms
from PIL import Image
import os
import sys

# Add root directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from models.encoder.encoder import EncoderCNN
from models.decoder.decoder import DecoderRNN
from models.fusion.model import ImageCaptioner
from tokenizer.tokenizer import Tokenizer
from dataset.flickr8k_handler import Flickr8kDataset

def run_inference(image_path, model_path="checkpoints/best/model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Tokenizer
    # We need to build the vocab exactly like we did in training
    # Alternatively, we could have saved/loaded the vocab, but for now we'll rebuild it from the token file
    tokenizer = Tokenizer()
    captions_file = "dataset/raw/flickr8k/Flickr8k.token.txt"
    with open(captions_file, 'r') as f:
        sentences = [line.strip().split('\t')[1] for line in f if len(line.strip().split('\t')) >= 2]
    tokenizer.build_vocab(sentences)
    vocab_size = len(tokenizer.vocab)

    # 2. Initialize Model
    embed_size = 256
    hidden_size = 512
    num_layers = 1
    
    encoder = EncoderCNN(embed_size)
    decoder = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers)
    model = ImageCaptioner(encoder, decoder).to(device)
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print("Model loaded successfully.")
    else:
        print(f"Error: Model not found at {model_path}")
        return

    # 3. Prepare Image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    # 4. Generate Caption
    with torch.no_grad():
        features = model.encoder(image_tensor)
        # The generate_caption method in model.py expects features and tokenizer
        # Let's use the one from the model class
        result_ids = model.generate_caption(image_tensor, tokenizer, device=device)
        
        # Convert IDs to words
        caption = []
        for word_id in result_ids:
            word = tokenizer.inv_vocab.get(word_id, "<UNK>")
            if word == "<END>":
                break
            if word not in ["<START>", "<PAD>"]:
                caption.append(word)
        
        generated_text = " ".join(caption)
        print(f"\nGenerated Caption: {generated_text}")
        return generated_text

if __name__ == "__main__":
    if len(sys.argv) > 1:
        img_p = sys.argv[1]
    else:
        # Default to a random image from the dataset if none provided
        img_p = "dataset/raw/flickr8k/Images/69189650_6687da7280.jpg"
    
    run_inference(img_p)