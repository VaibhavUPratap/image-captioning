import torch
import torch.nn as nn
import torchvision.models as models

class EncoderCNN(nn.Module):
    def __init__(self, embed_size, train_cnn=False):
        super(EncoderCNN, self).__init__()
        self.train_cnn = train_cnn
        
        # Load a pre-trained ResNet-50
        resnet = models.resnet50(pretrained=True)
        
        # Remove the last fully connected layer (classification layer)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        
        # Linear layer to map CNN features to embed_size
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, images):
        # Extract features
        with torch.no_grad() if not self.train_cnn else torch.enable_grad():
            features = self.resnet(images)
        
        # Flatten and embed
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        features = self.relu(features)
        features = self.dropout(features)
        
        return features

if __name__ == "__main__":
    # Test
    embed_size = 256
    encoder = EncoderCNN(embed_size)
    images = torch.randn(1, 3, 224, 224)
    output = encoder(images)
    print(f"Encoder Output shape: {output.shape}") # Should be [1, 256]
