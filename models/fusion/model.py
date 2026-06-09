import torch
import torch.nn as nn

class ImageCaptioner(nn.Module):
    def __init__(self, encoder, decoder):
        super(ImageCaptioner, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs

    def generate_caption(self, image, tokenizer, max_len=20, device='cpu'):
        """
        Inference: Generate a caption for a single image using greedy search.
        """
        result_caption = []
        
        with torch.no_grad():
            features = self.encoder(image) # [1, embed_size]
            states = None
            
            # Initial input is just the features
            inputs = features.unsqueeze(1) # [1, 1, embed_size]
            
            for i in range(max_len):
                hiddens, states = self.decoder.lstm(inputs, states)
                outputs = self.decoder.linear(hiddens.squeeze(1))
                predicted = outputs.argmax(1) # Greedy choice
                
                word_idx = predicted.item()
                result_caption.append(word_idx)
                
                # If <END> token is predicted, stop
                if word_idx == tokenizer.end_idx:
                    break
                
                # Prepare next input
                inputs = self.decoder.embedding(predicted).unsqueeze(1)
                
        return result_caption

if __name__ == "__main__":
    from models.encoder.encoder import EncoderCNN
    from models.decoder.decoder import DecoderRNN
    
    embed_size = 256
    hidden_size = 512
    vocab_size = 5000
    
    encoder = EncoderCNN(embed_size)
    decoder = DecoderRNN(embed_size, hidden_size, vocab_size)
    model = ImageCaptioner(encoder, decoder)
    
    print("Full Image Captioning Model initialized.")
