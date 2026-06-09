import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu
import os
import sys

# Ensure root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from dataset.flickr8k_handler import Flickr8kDataset
from tokenizer.tokenizer import Tokenizer
from models.encoder.encoder import EncoderCNN
from models.decoder.decoder import DecoderRNN
from models.fusion.model import ImageCaptioner

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on: {device}")

    # 1. Setup
    root_dir = "dataset/raw/flickr8k/Images"
    captions_file = "dataset/raw/flickr8k/Flickr8k.token.txt"
    test_split_file = "dataset/raw/flickr8k/Flickr_8k.testImages.txt"
    model_path = "checkpoints/best/model.pth"
    
    # Load test image IDs
    with open(test_split_file, 'r') as f:
        test_ids = set([line.strip() for line in f if line.strip()])

    # 2. Tokenizer
    tokenizer = Tokenizer()
    with open(captions_file, 'r') as f:
        sentences = [line.strip().split('\t')[1] for line in f if len(line.strip().split('\t')) >= 2]
    tokenizer.build_vocab(sentences)
    vocab_size = len(tokenizer.vocab)

    # 3. Model
    embed_size = 256
    hidden_size = 512
    num_layers = 1
    
    encoder = EncoderCNN(embed_size)
    decoder = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers)
    model = ImageCaptioner(encoder, decoder).to(device)
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print("Model loaded.")
    else:
        print("Model not found!")
        return

    # 4. Prepare References
    # Map image id -> list of tokenized reference captions
    references_dict = {}
    with open(captions_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 2: continue
            
            img_id = parts[0].split('#')[0]
            # Clean image ID extension if needed
            if '.jpg' in img_id.lower() and not img_id.lower().endswith('.jpg'):
                img_id = img_id[:img_id.lower().find('.jpg') + 4]
            
            if img_id in test_ids:
                caption = parts[1]
                # Tokenize reference (lowercase and clean)
                ref_tokens = tokenizer._tokenize(caption)
                if img_id not in references_dict:
                    references_dict[img_id] = []
                references_dict[img_id].append(ref_tokens)

    # 5. Transform for images
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # 6. Generate and compare
    hypotheses = []
    references = []
    
    print(f"Generating captions for {len(references_dict)} test images...")
    
    # We use a simple loop over test_ids that we actually found references for
    valid_test_ids = list(references_dict.keys())
    
    for img_id in tqdm(valid_test_ids):
        img_path = os.path.join(root_dir, img_id)
        if not os.path.exists(img_path):
            continue
            
        # Load image
        from PIL import Image
        image = Image.open(img_path).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)
        
        # Generate caption
        with torch.no_grad():
            result_ids = model.generate_caption(image, tokenizer, device=device)
            # Convert to tokens
            tokens = []
            for idx in result_ids:
                word = tokenizer.inv_vocab.get(idx, "<UNK>")
                if word == "<END>": break
                if word not in ["<START>", "<PAD>"]:
                    tokens.append(word)
            
            hypotheses.append(tokens)
            references.append(references_dict[img_id])

    # 7. Calculate BLEU scores
    print("\n" + "="*30)
    print("Captions Evaluation Scores:")
    print(f"BLEU-1: {corpus_bleu(references, hypotheses, weights=(1, 0, 0, 0)):.4f}")
    print(f"BLEU-2: {corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0)):.4f}")
    print(f"BLEU-3: {corpus_bleu(references, hypotheses, weights=(0.33, 0.33, 0.33, 0)):.4f}")
    print(f"BLEU-4: {corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25)):.4f}")
    print("="*30)

if __name__ == "__main__":
    evaluate()
