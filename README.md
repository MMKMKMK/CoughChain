# 🌐 CoughChain: An Environmentally Adaptive Cough Detection Framework Guided by Diverse Chain-of-Thought Prompting in LLM

CoughChain is a **training-free**, large language model (LLM)-based cough detection framework that leverages **diverse chain-of-thought (CoT) prompting strategies** and **environment-aware context injection** to achieve robust and interpretable cough recognition in complex acoustic environments.

> 🔍 **Core Idea**: By explicitly informing the LLM whether the input audio comes from a *"quiet"* or *"noisy"* environment—and guiding its reasoning through structured CoT prompts—the model **adaptively adjusts its decision logic** without any fine-tuning or parameter updates.

---

## 📌 Key Highlights

- ✅ **Zero training required**: Uses off-the-shelf LLM (Qwen-Omni-Turbo) via prompt engineering only  
- 🧠 **Four CoT reasoning strategies**:
  - `audiofeature`: Audio feature–guided reasoning (e.g., spectral, energy, zero-crossing rate)
  - `step`: Linear, step-by-step logical decomposition
  - `selfask`: Iterative self-questioning and answering to refine judgment
  - `tot` (Tree of Thought): Multi-branch tree-like reasoning with path exploration and consensus
- 🌍 **Environment-adaptive**: Dynamically switches reasoning strategy based on acoustic context ("quiet" vs. "noisy")
- 🎯 Evaluated on a **balanced, real-world multi-scenario cough dataset**

---

## 📂 Code Structure

The repository contains three main directories, each corresponding to a specific experimental setting:

| Directory / File | Description |
|------------------|-------------|
| `cot/`           | **Baseline & four CoT implementations** using **Qwen-Omni-Turbo**:<br>• `audiofeature.py` – LLM reasoning guided by audio features<br>• `step.py` – Stepwise logical decomposition of cough judgment<br>• `selfask.py` – Self-questioning to iteratively focus on key evidence<br>• `tot.py` – Tree-based multi-path reasoning with final consensus |
| `quiet/`         | **Quiet-environment evaluation**: Runs CoughChain with `"quiet"` context injected into prompts |
| `noisy/`         | **Noisy-environment evaluation**: Runs CoughChain with `"noisy"` context to enhance robustness against interference |

> 💡 All implementations are **purely prompt-driven**—switching environments or reasoning styles only requires modifying the prompt template. **No model weights are altered.**

---

## ⚙️ Essential Setup Before Running

Before executing any script, complete the following two critical configurations:

### 1. Configure Your Qwen API Key
Obtain your **Qwen-Omni-Turbo API key** from the [DashScope Console](https://dashscope.console.aliyun.com/), then choose one of the following methods:

- **Option A: Hardcode in script (for quick testing)**
  ```python
  client = OpenAI(
      api_key="YOUR_API_KEY_HERE",  # ← Replace with your actual key
      base_url="https://dashscope.aliyuncs.com/..."
  )

### 2. Update the Audio Root Directory Path (❗Critical!)
In every Python script (e.g., cot/audiofeature.py, noisy/selfask.py, etc.), locate the following line:
  ```python
  root_audio_dir = "/path/to/your/cough_audio_dataset"  # ← MUST UPDATE!Replace it with the absolute or relative path to your local cough audio folder.
  ```

## 📊 Code and Data Availability

- 🔒 The **full source code** will be publicly released upon **official acceptance of the paper**.
- 📁 A **subset of the cough audio dataset** will be provided in this repository .
- 📩 For access to the **complete dataset** (including noisy-scenario recordings and metadata), please contact us via email at: **fmingkk@163.com**.

> We encourage the use of this data strictly for academic and non-commercial research purposes, in accordance with ethical guidelines.
