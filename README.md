# AAMLA
An Autonomous Memory-Efficient Parameterizable LLM-Aided Hardware Generation Framework
![merge](assets/merged.png)

Related papers/repos:

1. https://github.com/princeton-nlp/MeZO
2. https://github.com/taehokim20/LLMem
3. https://github.com/ehw-fit/evoapproxlib/tree/v2022
4. RocketPPA: Ultra-Fast LLM-Based PPA Estimator at Code-Level Abstraction (https://arxiv.org/pdf/2503.21971)
5. https://github.com/hkust-zhiyao/RTL-Coder

## Getting Started

### Installation
Set virtual environment
```bash
conda create -n aamla python=3.10
conda activate aamla
```

Install pytorch
```bash
pip install torch==2.6.0
```

Install requirements
```bash
git clone https://github.com/rajatbh21/AAMLA.git
cd AAMLA
pip install -r requirements.txt
```

Install requirements for measuring pass@k
```bash
sudo apt-get install -y jq bc
```
Install VCS
- VCS is a Verilog compiler required for automated testing on benchmarks. Follow these steps to install and configure VCS:

- Obtain VCS from Synopsys. Ensure you have the required license to use it.
- Install VCS following the instructions provided in the official Synopsys documentation.
- Add the VCS executable to your system's PATH environment variable.
- Verify the installation by running:

### Usage
1. Run AAMLA.
    ```bash
    python aamla.py
    ```
2. Choose model, batch size and sequence length.
    https://huggingface.co/models
3. You can check the expected memory depending on the fine-tuning method.
4. Choose prioritization category between accuracy and latency.
5. Training will start automatically based on your selection.