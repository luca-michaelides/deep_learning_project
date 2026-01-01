# Deep Learning Project

This repository contains experiments, scripts, and supporting code for working with **large language models (LLMs)**, with a particular focus on **LLaMA‑2** and related attack / analysis workflows.

The project is structured to be:

* reproducible,
* explicit about model downloads and licensing,
* safe for GitHub (large model weights are *not* committed), and
* easy to extend for research and experimentation.

---

## 📁 Repository Structure

```text
deep_learning_project/
│
├── llm-attacks/                 # Core codebase for LLM attacks & experiments
│   ├── api_experiments/
│   ├── data/
│   ├── experiments/
│   ├── llm_attacks/
│   ├── scripts/
│   ├── demo.ipynb
│   ├── README.md
│   ├── requirements.txt
│   └── setup.py
│
├── models/                      # Local model storage (gitignored)
│   └── llama2-7b-chat-hf/       # Downloaded LLaMA‑2 model files
│
├── scripts/                     # Project-level utility scripts
│   └── download_llama2.py       # Downloads LLaMA‑2 via Hugging Face
│
├── .gitignore
└── README.md                    # This file
```

> **Note**: The `models/` directory is intentionally excluded from version control.

---

## 🧠 Models

This project currently uses **LLaMA‑2‑7B‑Chat (HF format)**.

* Source: `meta-llama/Llama-2-7b-chat-hf`
* License: Meta LLaMA‑2 Community License

You **must**:

1. Have a Hugging Face account
2. Accept the LLaMA‑2 license on Hugging Face

Model weights are downloaded locally and **never committed to GitHub**.

---

## ⬇️ Downloading LLaMA‑2

A helper script is provided to download the model snapshot locally.

### 1️⃣ Set up environment

```bash
pip install -U transformers huggingface_hub torch accelerate
```

Log in to Hugging Face:

```bash
huggingface-cli login
```

### 2️⃣ Download the model

From the repository root:

```bash
python scripts/download_llama2.py
```

After completion, the model will be available at:

```text
models/llama2-7b-chat-hf/
```

---

## 🚀 Loading the Model (Offline)

Once downloaded, the model can be loaded entirely offline using `transformers`:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "./models/llama2-7b-chat-hf"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
    device_map="auto",  # or "cpu"
    torch_dtype="auto"
)
```

---

## ⚠️ Git & Large Files

Model weights are **not tracked** by Git.

Ensure the following is present in `.gitignore`:

```gitignore
models/
```

This keeps the repository lightweight and avoids pushing large binaries or licensed artifacts.

---

## 🧪 Experiments & Attacks

The `llm-attacks/` directory contains:

* prompt- and API-based experiments
* attack implementations
* datasets and experiment outputs
* notebooks for exploratory analysis

Refer to `llm-attacks/README.md` for details on specific experiments and usage.

---

## 🖥️ Hardware Notes

* **CPU-only** loading is supported (slow, high RAM usage)
* **GPU** recommended for inference
* 4‑bit / 8‑bit quantization supported via `bitsandbytes`

Example (4‑bit):

```python
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    load_in_4bit=True,
    device_map="auto"
)
```

---

## 📌 Reproducibility

To reproduce results:

1. Clone this repository
2. Set up the Python environment
3. Download the model using the provided script
4. Run experiments from `llm-attacks/`

All non-determinism is isolated to model inference where applicable.

---

## 📜 License

This repository contains **code only**.

* Code: MIT (unless otherwise specified)
* Models: governed by their respective licenses (e.g. LLaMA‑2 license)

You are responsible for complying with model licensing terms.

---

## ✨ Notes

* This repo intentionally avoids Git submodules for simplicity
* Model downloads are explicit and script-driven
* Structure is designed for research and experimentation, not production deployment

---

If you plan to extend this project (new models, attacks, or benchmarks), consider adding:

* a `Makefile`
* experiment configuration files
* structured logging and result tracking
