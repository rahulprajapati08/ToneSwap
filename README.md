# 🎙️ ToneSwap: Real-Time Voice Conversion App

**ToneSwap** is a real-time voice conversion application that transforms a speaker's voice into a target voice while preserving the speech content, tone, and emotion. It supports various audio formats and provides a user-friendly web interface via Gradio.

---

## 🚀 Features

- 🎤 Converts your voice to another person's voice using voice samples
- 🧠 Zero-shot voice conversion using pretrained models (FreeVC + WavLM + HiFi-GAN)
- 📁 Supports uploading or recording source and target audio in any format
- 🔁 Automatically converts input to mono 16kHz `.wav` using FFmpeg
- 🌐 Easy-to-use Gradio web interface

---

## 📦 Prerequisites

Before running the app, download and place the following files and models in the correct directories:

### ✅ 1. Pretrained Checkpoints

Download FreeVC model checkpoints and place them in:

checkpoints/

📥 [Download Checkpoints](https://1drv.ms/u/s!AnvukVnlQ3ZTx1rjrOZ2abCwuBAh?e=UlhRR5)

---

### ✅ 2. WavLM-Large Model

Download the WavLM-Large model files from the official Microsoft repository and place them in:

wavlm/

📥 [WavLM GitHub](https://github.com/microsoft/unilm/tree/master/wavlm)

---

### ✅ 3. HiFi-GAN Vocoder (Optional, for SR training or fine-tuning)

Clone the HiFi-GAN repository and download `generator_v1` pretrained model. Place it in:

hifigan/

📥 [HiFi-GAN GitHub](https://github.com/jik876/hifi-gan)

---

## 🧪 Setup Instructions

1. **Clone this repo:**
   ```bash
   git clone https://github.com/<your-username>/ToneSwap.git
   cd ToneSwap
2. Install Python dependencies:
    ```bash
    pip install -r requirements.txt

3. Install FFmpeg (for audio format conversion):
   Windows: [ FFmpeg Download](https://ffmpeg.org/download.html)
5. Install Gradio:
   ```bash
   pip install gradio

---
##✅ How to Run
  ```bash
    python app.py
---
## ⚠️ Note
Due to GitHub's file size limitations, this repository does not include pretrained models or checkpoints. You must download and place them manually as instructed above.

