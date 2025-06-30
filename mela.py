#!/usr/bin/env python3
import subprocess
import questionary
import sys
import torch
import pandas as pd

from pyfiglet import Figlet

offline_data = pd.DataFrame([
    {"method": "fft+lora",            "peak_mem": 46549, "accuracy": 39.87, "time": 27},
    {"method": "fft+dora",            "peak_mem": 47733, "accuracy": 41.09, "time": 32.5},
    {"method": "tokentune+none",      "peak_mem": float("inf"), "accuracy": float("inf"), "time": float("inf")},
    {"method": "tokentune+lora",      "peak_mem": 47395,  "accuracy": 17.82, "time": 18.8},
    {"method": "mezo+none",           "peak_mem": 22397, "accuracy": 19.04, "time": 13.6},
    {"method": "apollo+none",           "peak_mem": 47359,  "accuracy": 36.60, "time":  11},
])

COMBINATIONS = [
    ("fft", "lora"),
    ("fft", "dora"),
    ("tokentune", "none"),
    ("tokentune", "lora"),
    ("mezo", "none"),
    ("apollo", "none"),
]

key_to_method_name = {
    "fft+lora": "LoRA",
    "fft+dora": "DoRA",
    "tokentune+none": "TokenTune",
    "tokentune+lora": "TokenTune+LoRA",
    "mezo+none": "MeZO",
    "apollo+none": "APOLLO"
}

VRAM_MiB = torch.cuda.get_device_properties(0).total_memory / (1024**2) if torch.cuda.is_available() else None

def print_banner():
    f = Figlet(font='slant')
    banner = f.renderText('MELA')
    print(banner)
    if VRAM_MiB:
        print(f"Available GPU Memory: {VRAM_MiB:.2f} MiB")
    else:
        print("No GPU detected")

def estimate_memory(model, batch_size, seq_length, method, adapter):
    cmd = ["bash", "LLMEM/run_llmem.sh", model, str(batch_size), str(seq_length), method, adapter, "false"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    for line in result.stdout.splitlines()[::-1]:
        if "Estimated peak memory:" in line:
            try:
                mb_value = float(line.split(":")[1].strip().split()[0])
                return mb_value
            except Exception:
                continue
    print(f"Error: could not parse peak memory estimate for {method}+{adapter}")
    print("--- stdout ---")
    print(result.stdout)
    print("--- stderr ---")
    print(result.stderr)
    sys.exit(1)


def get_inputs():
    model = questionary.text("📥 Enter model name:").ask()
    batch_size = int(questionary.text("📥 Enter batch size:").ask())
    seq_length = int(questionary.text("📥 Enter sequence length:").ask())
    return model, batch_size, seq_length


def get_memory_estimates(model, batch_size, seq_length):
    estimates = {}
    for method, adapter in COMBINATIONS:
        key = f"{method}+{adapter}"
        mem = estimate_memory(model, batch_size, seq_length, method, adapter)
        estimates[key] = mem
    
        print(f"🔍 {key_to_method_name[key]}: Estimated Memory = {mem:.2f} MiB")
    return estimates


def choose_prioritization():
    return questionary.select(
        "⚖️ Choose prioritization category:",
        choices=["accuracy", "time", "🔙 Go back"]
    ).ask()


def select_method(estimates, priority):
    df = offline_data.copy()
    df['est_mem'] = df['method'].map(estimates)

    ascending = True if priority == "time" else False
    df = df.sort_values(by=priority, ascending=ascending)

    display_choices = []
    for _, row in df.iterrows():
        key = row['method']
        method_name = key_to_method_name.get(key, key)
        peak_mem = row['peak_mem']
        est_mem = row['est_mem']

        mark = '✅' if VRAM_MiB and est_mem <= VRAM_MiB else '❌'
        if est_mem > VRAM_MiB:
            continue
        label = (
            f"{mark} {method_name} | Est: {est_mem:.2f} | Peak: {peak_mem:.2f} MiB | "
            f"Acc: {row['accuracy']}% | Time: {row['time']}h"
        )
        display_choices.append(label)

    display_choices.append("🔙 Go back")

    pick = questionary.select("🚀 Select method to run:", choices=display_choices).ask()
    if pick == "🔙 Go back":
        return None

    # Get the key back from the method name
    for key, name in key_to_method_name.items():
        if name in pick:
            return key
    return None



def main():
    print_banner()

    step = 0
    model, batch_size, seq_length = None, None, None
    estimates = None
    selected_method = None

    while True:
        if step == 0:
            model, batch_size, seq_length = get_inputs()
            step = 1

        elif step == 1:
            estimates = get_memory_estimates(model, batch_size, seq_length)
            step = 2

        elif step == 2:
            priority = choose_prioritization()
            if priority == "🔙 Go back":
                step = 1
            else:
                step = 3
                selected_priority = priority

        elif step == 3:
            selected_method = select_method(estimates, selected_priority)
            if selected_method is None:
                step = 2
            else:
                method_display = key_to_method_name[selected_method]
                print(f"▶️ Starting fine-tuning with {method_display}...")

                if selected_method == 'fft+lora':
                    subprocess.run(["bash", "-c", "cd LLaMA-Factory && llamafactory-cli train examples/train_codellama/codellama_lora_sft.yaml"])

                elif selected_method == 'fft+dora':
                    subprocess.run(["bash", "-c", "cd LLaMA-Factory && llamafactory-cli train examples/train_codellama/codellama_dora_sft.yaml"])

                elif selected_method == 'apollo+none':
                    subprocess.run(["bash", "-c", "cd LLaMA-Factory && llamafactory-cli train examples/train_codellama/codellama_apollo_sft.yaml"])

                elif selected_method == 'mezo+none':
                    subprocess.run(["bash", "-c", "MODEL=codellama/CodeLlama-7b-Instruct-hf TASK=HaVen MODE=ft BS={} LR=1e-6 EPS=1e-4 bash MeZO/large_models/mezo.sh".format(batch_size)])

                elif selected_method in ['tokentune+none', 'tokentune+lora']:
                    subprocess.run(["bash", "tokentune/scripts/train/lora-tokentune-codellama.sh"])
                break


if __name__ == "__main__":
    main()