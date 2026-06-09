import gradio as gr
import torch
from torchvision import transforms
from PIL import Image
import os
import sys

# Ensure root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from models.encoder.encoder import EncoderCNN
from models.decoder.decoder import DecoderRNN
from models.fusion.model import ImageCaptioner
from tokenizer.tokenizer import Tokenizer

# --- Initialization ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Tokenizer & Vocab
tokenizer = Tokenizer()
captions_file = "dataset/raw/flickr8k/Flickr8k.token.txt"
if os.path.exists(captions_file):
    with open(captions_file, 'r') as f:
        sentences = [line.strip().split('\t')[1] for line in f if len(line.strip().split('\t')) >= 2]
    tokenizer.build_vocab(sentences)
else:
    print("Warning: Captions file not found. Tokenizer might not be correctly initialized.")

# Load Model
embed_size = 256
hidden_size = 512
vocab_size = len(tokenizer.vocab)
model_path = "checkpoints/best/model.pth"

encoder = EncoderCNN(embed_size)
decoder = DecoderRNN(embed_size, hidden_size, vocab_size, 1)
model = ImageCaptioner(encoder, decoder).to(device)

if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("Model loaded successfully.")
else:
    print(f"Error: Model not found at {model_path}")

# --- Prediction Function ---
def predict(inp_img):
    if inp_img is None:
        return ""
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    image = Image.fromarray(inp_img).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        # result_ids = model.generate_caption(image_tensor, tokenizer, device=device)
        # Using the logic from inference.py to ensure it returns string
        features = model.encoder(image_tensor)
        # Manual greedy search as defined in model.py generate_caption
        # but let's just use the model's method as it's cleaner
        result_ids = model.generate_caption(image_tensor, tokenizer, device=device)
        
        caption = []
        for word_id in result_ids:
            word = tokenizer.inv_vocab.get(word_id, "<UNK>")
            if word == "<END>": break
            if word not in ["<START>", "<PAD>"]:
                caption.append(word)
        
        return " ".join(caption)

# --- Gradio UI ---
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(),
    outputs=gr.Text(label="Generated Caption"),
    title="AI Image Captioning",
    description="Upload an image and the trained CNN-LSTM model will describe it for you."
)

if __name__ == "__main__":
    demo.launch(share=False)
