from datasets import load_dataset
import json
import os

dataset = load_dataset("yangyiyao/HaVen-KL-Dataset", split="train")

processed_data = []
for item in dataset:
    processed_data.append({
        "instruction": item["instruction"],
        "input": item.get("input", ""),
        "output": item["output"]
    })

save_path = "data/"
file_name = "haven_kl_data.json"
full_path = os.path.join(save_path, file_name)
os.makedirs(save_path, exist_ok=True)

with open(full_path, "w", encoding="utf-8") as f:
    json.dump(processed_data, f, ensure_ascii=False, indent=2)

print(f"Saved to {full_path}")