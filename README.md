# RETL-ASR-STT

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

A repository for **parameter-efficient fine‑tuning** of OpenAI Whisper models on speech recognition (ASR) and speech‑to‑text translation (STT) tasks using LoRA, LoReFT, and DiReFT methods, built on top of the [StanfordNLP pyreft](https://github.com/stanfordnlp/pyreft) package.

---

## 📂 Repository Structure

```text
RETl-ASR-STT/
├── pyreft/                 # Modified pyreft package modules (core library)
├── ASR/                    # Whisper ASR training & evaluation scripts
│   ├── full_params.py      # Full-model fine‑tuning baseline
│   ├── peft_reft.py        # LoRA, LoReFT, DiReFT implementations for ASR
│   ├── test_full_params.py # ASR evaluation for full-params-fietune
│   └── test_peft_reft.py   # ASR evaluation for Lora,Direft and Loreft
├── STT/                    # Whisper STT (speech translation) scripts
│   ├── full_params.py      # Full-model fine‑tuning baseline
│   ├── peft_reft.py        # LoRA, LoReFT, DiReFT implementations for STT
│   ├── test_full_params.py # STT evaluation for full-params-fietune
│   └── test_peft_reft.py   # STT evaluation for for Lora,Direft and Loreft
└── README.md               # Project overview and instructions
```

---

## 🚀 Installation

1. **Clone this repository**
   ```bash
   git clone https://github.com/makabala-sss/RETL-ASR-STT.git
   cd RETL-ASR-STT
   ```

2. **Install Python dependencies**
   ```bash
   pip install torch transformers datasets jiwer sacremoses tqdm
   ```

3. **Install `pyreft` package** (modified from StanfordNLP):
   ```bash
   # In the repository root
   pip install -e pyreft
   ```

4. **Follow the original pyreft tutorial for additional setup**:
   See [StanfordNLP pyreft Installation & Usage](https://github.com/stanfordnlp/pyreft) for detailed instructions on environment configuration, required tools, and dataset preparation.

---

## 💡 Usage

### 1. Automatic Speech Recognition (ASR)

Navigate into the `ASR/` folder to fine‑tune and evaluate Whisper models for speech recognition:

```bash
cd ASR
# Example: full-model fine-tuning on your dataset
python full_params.py --model_size small --use_fp16 False

# Example: LoRA / LoReFT / DiReFT fine-tuning
python peft_reft.py --model_size medium --intervention Loreft --reft_r 4
python peft_reft.py --model_size large --intervention Lora --lora_r 8 --lora_alpha 16

# Evaluate
python test_peft_reft.py --model_size medium --intervention Loreft --reft_r 4  --base_dir ./checkpoints
```

### 2. Speech‑to‑Text Translation (STT)

Navigate into the `STT/` folder to fine‑tune and evaluate Whisper models for speech translation:

```bash
cd STT
# Example: full-model fine-tuning
python full_params.py --model_size small --use_fp16 False

# Example: LoRA / LoReFT / DiReFT
python peft_reft.py --model_size large --intervention Lora --lora_r 8 --lora_alpha 16

# Evaluate
python test_peft_reft.py --model_size medium --intervention Lora --lora_r 8 --lora_alpha 16  --base_dir ./checkpoints
```

**Options**:
- `--model_size`: one of `small`, `medium`, `large`
- `--method`: one of `lora`, `loreft`, `direft`
- Additional args: see script docstrings (`-h` for help).

---

## 🔧 Key Files

- **`pyreft/`**: Core intervention library adapted from [stanfordnlp/pyreft](https://github.com/stanfordnlp/pyreft). Contains tokenization, dataset, intervention and trainer implementations.
- **`ASR/peft_reft.py`** & **`STT/peft_reft.py`**: Entry points for parameter-efficient fine‑tuning methods (LoRA, LoReFT, DiReFT).
- **`ASR/full_params.py`** & **`STT/full_params.py`**: Full-model fine-tuning baselines for comparison.
- **Test scripts**: `test_full_params.py` & `test_peft_reft.py` in both `ASR/` and `STT/` folders evaluate WER (ASR) and BLEU (STT).

---

## 📚 References & Links

- **pyreft (StanfordNLP)**: https://github.com/stanfordnlp/pyreft
- **OpenAI Whisper**: https://github.com/openai/whisper
- **Hugging Face Transformers**: https://github.com/huggingface/transformers
- **Datasets library**: https://github.com/huggingface/datasets

---

## 🤝 Contributing

Contributions and issues are welcome! Please open a GitHub issue or submit a pull request with a clear description of your changes.

---

## 📄 License

This project is licensed under the Apache License, Version 2.0. See the [LICENSE](LICENSE) file for details.
