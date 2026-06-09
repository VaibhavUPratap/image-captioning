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
- `Flickr_8k.testImages.txt` (Test split list)

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
│   ├── flickr8k_handler.py # Robust data loading & cleaning
│   └── download_data.py    # (Optional) Helper for data
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

## 📝 License
MIT
