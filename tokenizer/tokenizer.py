import os
import json
from collections import Counter
import re

class Tokenizer:
    def __init__(self, vocabulary=None, max_vocab_size=10000, min_freq=2):
        self.vocab = vocabulary if vocabulary else {}
        self.max_vocab_size = max_vocab_size
        self.min_freq = min_freq
        
        self.specials = ["<PAD>", "<START>", "<END>", "<UNK>"]
        self.pad_idx = 0
        self.start_idx = 1
        self.end_idx = 2
        self.unk_idx = 3

        if not self.vocab:
            self._init_special_tokens()
        
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    def _init_special_tokens(self):
        for i, token in enumerate(self.specials):
            self.vocab[token] = i

    def build_vocab(self, sentences):
        counter = Counter()
        for sentence in sentences:
            tokens = self._tokenize(sentence)
            counter.update(tokens)
        
        # Filter by frequency and max size
        most_common = [word for word, count in counter.most_common(self.max_vocab_size) if count >= self.min_freq]
        
        for i, word in enumerate(most_common):
            if word not in self.vocab:
                idx = len(self.vocab)
                self.vocab[word] = idx
        
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        print(f"Vocabulary built. Size: {len(self.vocab)}")

    def _tokenize(self, text):
        # Basic alphanumeric tokenization
        text = text.lower()
        text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
        return text.split()

    def encode(self, text, add_special=True):
        tokens = self._tokenize(text)
        ids = [self.vocab.get(token, self.unk_idx) for token in tokens]
        
        if add_special:
            ids = [self.start_idx] + ids + [self.end_idx]
        
        return ids

    def decode(self, ids):
        tokens = [self.inv_vocab.get(idx, "<UNK>") for idx in ids]
        # Filter out special tokens for clean output
        clean_tokens = [t for t in tokens if t not in self.specials]
        return " ".join(clean_tokens)

    def save(self, filepath):
        with open(filepath, 'w') as f:
            json.dump(self.vocab, f)

    @classmethod
    def load(cls, filepath):
        with open(filepath, 'r') as f:
            vocab = json.load(f)
        return cls(vocabulary=vocab)

if __name__ == "__main__":
    # Test
    sentences = ["A dog runs in the grass.", "A cat sleeps on the sofa.", "The dog is happy."]
    tokenizer = Tokenizer()
    tokenizer.build_vocab(sentences)
    
    encoded = tokenizer.encode("A dog sleeps in the grass.")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {tokenizer.decode(encoded)}")
