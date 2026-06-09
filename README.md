# Image Captioning AI 📸🤖

An end-to-end Image Captioning system built with **PyTorch**, featuring a **CNN-LSTM** architecture, **Gradio** web interface, and comprehensive evaluation metrics.

## ✨ Features
- **Architecture**: ResNet-50 Encoder + LSTM Decoder.
- **Interactive UI**: Upload images and get captions via Gradio.
- **Robust Data Handling**: Custom Flickr8k dataset handler with automated missing-file filtering.
- **Evaluation**: BLEU score calculation (BLEU-1 to BLEU-4) for quality assessment.
- **Interactive Notebook**: Complete walkthrough of the entire pipeline.

---

## 🚀 Quick Start

### 1. Setup Environment
```bash
python -m venv myenv
source myenv/Scripts/activate  # Windows
pip install -r requirements.txt
```

### 2. Download Data
Place the Flickr8k dataset in `dataset/raw/flickr8k/`:
- `Images/` (Folder containing JPGs)
- `Flickr8k.token.txt` (Captions file)

### 3. Run Interactive UI
Launch the Gradio web app to upload and describe images:
```bash
python app.py
```
Visit `http://127.0.0.1:7860` in your browser.

---

## 🏗️ Project Structure
```text
image-captioning/
├── app.py                # Gradio Web Interface
├── image_captioning.ipynb # Interactive Notebook Walkthrough
├── training/train.py     # Main training loop
├── evaluate.py           # BLEU score evaluation script
├── inference.py          # Standalone prediction script
├── dataset/
│   └── flickr8k_handler.py # Robust data loading & cleaning
├── models/
│   ├── encoder/encoder.py  # ResNet-50 based Encoder
│   ├── decoder/decoder.py  # LSTM based Decoder
│   └── fusion/model.py     # ImageCaptioner combined model
└── tokenizer/tokenizer.py   # Text tokenization & vocab building
```

---

## 📊 Performance
The model was trained for 5 epochs on the Flickr8k dataset using an NVIDIA RTX 3050.

| Metric | Score |
| :--- | :--- |
| **BLEU-1** | **0.5138** |
| **BLEU-2** | **0.3305** |
| **BLEU-3** | **0.2067** |
| **BLEU-4** | **0.1294** |

---

## 🧪 Evaluation & Inference
- **Evaluate**: Run `python evaluate.py` to calculate BLEU scores on the test split.
- **Inference**: Run `python inference.py path/to/image.jpg` for a single prediction.


## 2) Development Roadmap

### Phase 1: Environment & Data Preparation
- [ ] Set up Python environment (requirements.txt/environment.yml).
- [ ] Implement data downloading scripts for COCO/Flickr.
- [ ] Build the preprocessing pipeline (Image resizing, Text normalization).
- [ ] Implement the Tokenizer.

### Phase 2: Model Architecture
- [ ] Implement the Encoder (typically a pre-trained CNN/Transformer).
- [ ] Implement the Decoder (RNN/Transformer-based).
- [ ] Define the Fusion layer/Attention mechanism.

### Phase 3: Training & Evaluation
- [ ] Write the training loops with logging (Tensorboard/Weights & Biases).
- [ ] Implement Loss functions (CrossEntropy, etc.).
- [ ] Implement Evaluation metrics (BLEU, METEOR, ROUGE, CIDEr).

### Phase 4: Inference & Deployment
- [ ] Implement Greedy and Beam Search decoding.
- [ ] Create a FastAPI/Flask application for model serving.
- [ ] Dockerize the application.

## 3) Technical Stack (Initial Selection)
- **Deep Learning:** PyTorch / TensorFlow
- **Computer Vision:** OpenCV / Pillow / Torchvision
- **NLP:** HuggingFace Tokenizers / NLTK
- **Monitoring:** TensorBoard
- **API:** FastAPI
- **Containerization:** Docker
- **utilities/**: shared concerns (device, reproducibility, I/O, experiment IDs).
- **configuration/**: centralized hyperparameters/environment controls.
- **notebooks/**: exploratory analysis and debugging, not production execution.
- **deployment/**: serving architecture and environment packaging plans.
- **api/**: request/response schema, validation and interface behavior.
- **docs/**: user/developer documentation artifacts.
- **tests/**: behavioral and quality validation of components/pipelines.
- **checkpoints/**: persisted model state for resume/best-model restoration.
- **logs/**: traceability and experiment observability.

## 2) Overall System Architecture

```text
Input Image
   ↓
Image Preprocessing (resize, normalize, augment for train only)
   ↓
CNN Encoder (visual feature extraction)
   ↓
Feature Projection / Embedding Space Alignment
   ↓
Transformer Decoder (language generation conditioned on image)
   ↓
Token Sequence
   ↓
Detokenization + Text Cleanup
   ↓
Final Caption
```

Data flow summary:
1. Image is standardized into model-ready tensor shape/distribution.
2. Encoder transforms pixels into semantically meaningful visual vectors.
3. Projected visual embeddings become memory for decoder cross-attention.
4. Decoder predicts caption token-by-token autoregressively.
5. Generated token IDs are mapped back to words and cleaned.

## 3) CNN Encoder Design

- **Responsibilities**: extract hierarchical visual representations from image regions/objects/context.
- **Input**: normalized image tensor (fixed height/width/channels).
- **Output**: dense feature map or pooled feature vectors in a shared embedding dimension.
- **Conceptual internals**:
  - early layers: edges, texture, corners
  - mid layers: patterns, object parts
  - deep layers: object-level and scene semantics
- **Feature extraction process**: convolutional hierarchy + nonlinearity + downsampling encodes increasing receptive field context.
- **Embedding generation**: final encoder representation is projected to decoder-compatible dimension to enable cross-attention.

## 4) Tokenization Module

- **Vocabulary creation**: build from training captions with frequency thresholding.
- **Special tokens**: `<pad>`, `<bos>`, `<eos>`, `<unk>`.
- **Mappings**:
  - word → index for model input
  - index → word for generation output
- **Padding**: sequences are padded to max length per batch (or global cap).
- **Sequence lengths**: track actual lengths for masking and loss calculation.
- **Text preprocessing**: lowercase policy, punctuation handling, whitespace normalization, optional number normalization.

Example concept:
- Caption: “A dog runs in grass”
- Tokenized: `<bos> a dog runs in grass <eos>`
- Indexed via vocabulary and padded for batch alignment.

## 5) Transformer Decoder Design

```text
Previous Tokens
   ↓
Token Embeddings + Positional Encoding
   ↓
Masked Self-Attention (cannot see future tokens)
   ↓
Cross-Attention (queries text, keys/values image features)
   ↓
Feed-Forward Network
   ↓
Output Projection to Vocabulary Logits
   ↓
Next Token Prediction
```

- **Embeddings**: convert token IDs to dense vectors.
- **Positional encoding**: inject token order information.
- **Masked self-attention**: preserves autoregressive generation.
- **Cross-attention**: aligns linguistic context with visual memory.
- **FFN blocks**: enrich representation capacity between attention layers.
- **Output projection**: maps hidden states to vocabulary probability space.

## 6) Training Pipeline

1. Load dataset samples (image path + one/more captions).
2. Apply train-time image transforms.
3. Preprocess and tokenize captions.
4. Build batches with masks (padding + causal masks).
5. Use teacher forcing:
   - decoder input: shifted ground-truth prefix
   - target: next token sequence
6. Forward pass through encoder + decoder.
7. Compute cross-entropy loss over non-pad positions.
8. Backpropagate gradients.
9. Optimizer update (and scheduler step if enabled).
10. Track train metrics and periodic validation metrics.
11. Save checkpoints (last, best-by-metric, periodic).
12. Early stopping/selection based on validation trend.

## 7) Inference Pipeline

```text
User Image Upload
   ↓
Preprocess Image
   ↓
CNN Feature Extraction
   ↓
Initialize Decoder with <bos>
   ↓
Iterative Token Generation
   ↓
Stop at <eos> or max length
   ↓
Detokenize and format caption
```

- **Greedy decoding**: pick highest-probability next token each step (fast, less diverse).
- **Beam search**: maintain top-k candidate sequences with cumulative scores (better quality, more compute).

## 8) Dataset Organization

Recommended structure:

```text
dataset/
├── raw/
│   ├── coco/{images,annotations}
│   ├── flickr8k/{images,captions}
│   └── flickr30k/{images,captions}
├── processed/
│   ├── images/
│   ├── captions_cleaned/
│   └── tokenizer_assets/
└── splits/
    ├── train_manifest
    ├── val_manifest
    └── test_manifest
```

Guidelines:
- Keep raw datasets immutable.
- Store cleaned captions and resized/cached images separately.
- Split manifests should map each sample to image path and caption IDs.

## 9) Configuration System

Configurable items:
- learning rate, batch size, epochs
- embedding dimension, hidden dimension, number of layers/heads
- vocabulary size / frequency threshold
- image size and preprocessing policy
- optimizer and scheduler type/parameters
- checkpoint directory, save frequency
- device preference (CPU/GPU/MPS)
- random seed and reproducibility flags

Organization:
- `configuration/defaults/`: baseline settings
- `configuration/experiments/`: experiment-specific overrides
- `configuration/environments/`: machine/runtime-specific overrides
- `configuration/config_schema.md`: validation rules and required keys

## 10) Evaluation Module

- **BLEU**: n-gram precision overlap with references (good for lexical overlap).
- **METEOR**: harmonic balance with stemming/synonym matching (better linguistic flexibility).
- **ROUGE**: recall-oriented overlap (useful for content coverage).
- **CIDEr**: consensus-based metric weighted by TF-IDF n-grams (strong for caption benchmarks).
- **SPICE**: scene-graph semantic matching (captures meaning/relations).

Usage recommendation:
- Track multiple metrics together; no single metric captures all caption quality aspects.
- Pair automatic metrics with manual qualitative review.

## 11) Deployment Structure

Two serving options:
- **FastAPI**: async-friendly, automatic schema docs, high-performance API use cases.
- **Flask**: lightweight, simple synchronous deployment.

Deployment responsibilities:
- model bootstrap/loading on service start
- upload/input validation and image decoding
- preprocessing and inference invocation
- decoding/postprocessing and response formatting
- health/readiness endpoint and error handling

## 12) Logging and Checkpoints

- **logs/training/**: epoch loss, learning rate, gradient stats.
- **logs/evaluation/**: metric snapshots by dataset split.
- **logs/tensorboard/**: scalar curves, optional attention/embedding visuals.
- **checkpoints/experiments/**: chronological snapshots.
- **checkpoints/best/**: top model by selected validation metric.
- Include experiment ID, config hash, and timestamp for traceability.

## 13) Documentation Plan

- **README**: project purpose, scope, high-level architecture.
- **INSTALLATION.md**: environment setup and dependency strategy.
- **ARCHITECTURE.md**: encoder/decoder/dataflow deep dive.
- **DATASET_GUIDE.md**: acquisition, licensing notes, preprocessing and splits.
- **TRAINING_GUIDE.md**: training lifecycle, configuration knobs, checkpoint resume policy.
- **INFERENCE_GUIDE.md**: decoding modes, runtime behavior, output formatting.
- **DEPLOYMENT_GUIDE.md**: API design, packaging, scalability and monitoring considerations.

## 14) Development Roadmap

- **Phase 1: Data + Tokenizer Foundation**
  - dataset ingestion, cleaning, split manifests, vocabulary policy
- **Phase 2: CNN Encoder**
  - visual backbone design and embedding projection contract
- **Phase 3: Transformer Decoder**
  - autoregressive language module with cross-attention
- **Phase 4: Integrated Model**
  - end-to-end encoder-decoder connectivity and masking correctness
- **Phase 5: Training System**
  - training loop, optimization, checkpointing, validation control
- **Phase 6: Evaluation System**
  - metric computation and benchmark reporting workflow
- **Phase 7: Inference Productization**
  - generation strategies (greedy/beam) and stable inference pipeline
- **Phase 8: Deployment**
  - API serving, packaging, runtime monitoring and reliability

## 15) Future Improvements

- stronger beam search strategies and length normalization
- attention map visualization for interpretability
- Vision Transformer encoder alternatives
- CLIP-style pretraining or multimodal contrastive warm-start
- multilingual caption generation
- text-to-speech for spoken captions
- voice-driven image captioning interaction
- domain-specific fine-tuning workflows
- reinforcement learning from human feedback for caption preference optimization
- mobile/on-device inference optimization
- real-time captioning for streaming inputs

---

This blueprint is intentionally **code-free** and intended to be used as a professional implementation guide for subsequent development.
