import torch
import torch.nn as nn

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        
        # Word embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_size)
        
        # LSTM layer
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        
        # Linear layer to map hidden state to vocabulary
        self.linear = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, features, captions):
        # Remove the last token (<END>) from captions for input
        # features shape: [batch_size, embed_size]
        # captions shape: [batch_size, seq_len]
        
        embeddings = self.embedding(captions[:, :-1])
        
        # Concatenate features and embeddings
        # features needs a sequence dimension: [batch_size, 1, embed_size]
        embeddings = torch.cat((features.unsqueeze(1), embeddings), dim=1)
        
        # LSTM output
        hiddens, _ = self.lstm(embeddings)
        
        # Map to vocabulary size
        outputs = self.linear(hiddens)
        
        return outputs

if __name__ == "__main__":
    # Test
    embed_size = 256
    hidden_size = 512
    vocab_size = 5000
    
    decoder = DecoderRNN(embed_size, hidden_size, vocab_size)
    
    # Mock data
    features = torch.randn(1, embed_size)
    captions = torch.randint(0, vocab_size, (1, 10))
    
    output = decoder(features, captions)
    print(f"Decoder Output shape: {output.shape}") # Should be [1, 10, 5000]
