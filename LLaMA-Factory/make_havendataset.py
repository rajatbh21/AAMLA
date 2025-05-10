from datasets import load_dataset
import json

dataset = load_dataset("yangyiyao/HaVen-KL-Dataset", split="train")

processed_data = []
for item in dataset:
    processed_data.append({
        "instruction": item["instruction"],
        "input": item.get("input", ""),
        "output": item["output"]
    })

output_path = "data/haven_kl_data.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(processed_data, f, ensure_ascii=False, indent=2)

print(f"Saved to {output_path}")