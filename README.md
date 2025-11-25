
# AAMLA  
## An Autonomous Memory-Efficient Parameterizable LLM-Aided Hardware Generation Framework

<p align="center">
  <img src="aamla.png" width="420"/>
</p>

[📄 **Paper**](https://www.techrxiv.org/users/948105/articles/1317868-aamla-an-autonomous-agentic-framework-for-memory-aware-llm-aided-hardware-generation)

AAMLA is an **LLM‑driven, memory‑aware hardware generation framework** that combines:

- Parameter‑efficient tuning (MeZO, LoRA, LLMem++)  
- LLM‑based RTL generation (RTL‑Coder, RocketPPA)  
- Approximate arithmetic exploration (EvoApprox)  
- Automated HW/SW co‑design loops  

---

# Getting Started

## 1. Environment Setup

```bash
conda create -n aamla python=3.10
conda activate aamla
pip install torch==2.6.0
```

Clone repository:

```bash
git clone https://github.com/rajatbh21/AAMLA.git
cd AAMLA
pip install -r requirements.txt
```

---

## 2. pass@k Tools (Optional)

```bash
sudo apt-get install -y jq bc
```

---

## 3. VCS Installation (for Verilog Testing)

1. Obtain from Synopsys  
2. Install + license  
3. Add to PATH  
4. Validate with:

```bash
vcs -ID
```

---

# Usage

Run AAMLA:

```bash
python aamla.py
```

![merge](assets/merged.png)

You will interactively select:

- Model (HuggingFace)
- Batch size / Sequence length
- Memory‑efficient fine‑tuning scheme  
- Priority: **Accuracy** or **Latency**  

Training + hardware generation starts automatically.

---

# Features

- Memory‑efficient LLM FT (MeZO, LoRA, LLMem++)
- Autonomous approximate computing search
- RTL generation + functional verification
- PPA estimation (area, power, delay)
- pass@k scoring for code quality
- End‑to‑end hardware design automation
