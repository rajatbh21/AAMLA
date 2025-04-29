from datasets import load_dataset
from transformers import AutoTokenizer
import matplotlib.pyplot as plt

def check_output_token():
    dataset = load_dataset("yangyiyao/HaVen-KL-Dataset")
    tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-350M-multi", use_fast=True)
    examples = dataset["train"]
    numoftoken = dict()
    over1024=0

    for idx, example in enumerate(examples):
        instruction = example['instruction']
        # responses = example['instruction']
        responses = example['output']

        # response는 리스트로 되어 있으므로 마지막 응답만 사용
        if not responses:
            continue  # 비어있는 응답은 무시

        responses += "\n<|endofcode|>"
        num = len(tokenizer.encode(responses, add_special_tokens=False))
        if numoftoken.get(num):
            numoftoken[num]+=1
        else:
            numoftoken[num]=1
        if num > 1024:
            over1024 += 1

    x = sorted(numoftoken.keys())              # 토큰 수
    y = [numoftoken[k] for k in x]             # 각 토큰 수에 대한 개수

    # 시각화
    plt.figure(figsize=(10, 6))
    plt.bar(x, y)
    plt.xlabel("Number of Tokens")
    plt.ylabel("Count")
    plt.title("Token Length Distribution")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    print(f"max number of token: {max(numoftoken)}, more than 1024 : {over1024}")

check_output_token()