# MELA
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
conda create -n mela python=3.10
conda activate mela
```

Install pytorch
```bash
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu118
```

Install requirements
```bash
git clone https://github.com/rajatbh21/MELA.git
cd MELA
pip install -r requirements.txt
```

Install requirements for measuring pass@k
```bash
sudo apt install autoconf
sudo apt install gperf
sudo apt install flex
sudo apt install bison

git clone https://github.com/steveicarus/iverilog.git
cd iverilog
git checkout v12-branch
sh autoconf.sh
./configure
make -j4
make install
```


### Usage
1. Run MELA.
    ```bash
    python mela.py
    ```
2. Choose model, batch size and sequence length.
    https://huggingface.co/models
3. You can check the expected memory depending on the fine-tuning method.
4. Choose prioritization category between accuracy and latency.
5. Training will start automatically based on your selection.