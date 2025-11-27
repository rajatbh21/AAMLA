
<p align="center">
  <img src="assets/aamla.png" width="420"/>
</p>


<div align="center"><h1>&nbsp;AAMLA: An Autonomous Agentic Framework for Memory-Aware LLM-Aided Hardware Generation</h1></div>



<p align="center">
 <a href="https://www.techrxiv.org/doi/full/10.36227/techrxiv.175393689.97544984"><b>Preprint</b></a> 
</p>

## Contents
- [News](#news)
- [Introduction](#introduction)
- [Working_with_AAMLA](#Working_with_AAMLA)
- [Installation](#installation)
- [Usage](#usage)
- [Citation](#citation)

## News
- [2025/07] AAMLA preprint is released.
- [2025/10] AAMLA is accepted at VLSID 2026.

## Introduction

### AAMLA: An Autonomous Agentic Framework for Memory-Aware LLM-Aided Hardware Generation

This repository accompanies the paper **“AAMLA: An Autonomous Agentic Framework for Memory-Aware LLM-Aided Hardware Generation,”** accepted at **VLSID 2026**. AAMLA is a novel framework that enables hardware designers to fine-tune LLMs on domain-specific hardware corpora while avoiding Out-of-Memory (OoM) failures on commodity GPUs. AAMLA supports a diverse suite of parameter- and memory-efficient fine-tuning techniques, automatically estimates memory requirements for a given model–dataset–method combination, and adaptively selects feasible configurations to ensure reliable, low-latency fine-tuning even under tight hardware budgets.

## Working_with_AAMLA

### 1. Environment Setup

```bash
conda create -n aamla python=3.10
conda activate aamla
pip install torch==2.6.0
```

### Clone repository:

```bash
git clone https://github.com/rajatbh21/AAMLA.git
cd AAMLA
pip install -r requirements.txt
```

---

### 2. pass@k Tools (Optional)

```bash
sudo apt-get install -y jq bc
```

---

### 3. VCS Installation (for Verilog Testing)

VCS is a Verilog compiler required for automated testing on benchmarks. Follow these steps to install and configure VCS:
1. Obtain VCS from Synopsys. Ensure you have the required license to use it.
2. Install VCS following the instructions provided in the official Synopsys documentation.
3. Add the VCS executable to your system's PATH environment variable.


Verify the installation by running:
```bash
vcs -help
```

---

## Usage

Run AAMLA:

```bash
python aamla.py
```

You will interactively select:

- Model (HuggingFace)
- Batch size / Sequence length
- Memory‑efficient fine‑tuning scheme (you are welcome to add more) 
- Priority: **Accuracy** or **Latency**  

The job starts.
---

## Dataset

For dataset, we took the [HaVen-KL-Dataset](https://huggingface.co/datasets/yangyiyao/HaVen-KL-Dataset). But the dataset is configurable.
