#!/usr/bin/env python3
import subprocess
import questionary
import sys
import torch
import pandas as pd

from pyfiglet import Figlet

offline_data = pd.DataFrame([
    {"method": "fft+lora",            "peak_mem": 11, "accuracy": 0.85, "time": 120},
    {"method": "fft+dora",            "peak_mem": 10, "accuracy": 0.83, "time": 110},
    {"method": "tokentune+none",      "peak_mem": 11, "accuracy": 0.85, "time": 120},
    {"method": "tokentune+lora",      "peak_mem": 8,  "accuracy": 0.80, "time": 100},
    {"method": "mezo+none",           "peak_mem": 8, "accuracy": 0.85, "time": 120},
    {"method": "apollo+none",           "peak_mem": 5,  "accuracy": 0.79, "time":  90},
])

COMBINATIONS = [
    ("fft", "lora"),
    ("fft", "dora"),
    ("tokentune", "none"),
    ("tokentune", "lora"),
    ("mezo", "none"),
    ("apollo", "none"),
]

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
        if method == 'fft':
            key.replace('fft', 'full fine-tuning')
        print(f"🔍 {key}: Estimated Memory = {mem:.2f} MiB")
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
        mark = '✅' if VRAM_MiB and row['est_mem'] <= VRAM_MiB else '❌'
        label = f"{mark} {row['method']} | Mem: {row['est_mem']:.2f}GB | Acc: {row['accuracy']} | Time: {row['time']}s"
        display_choices.append(label)
    display_choices.append("🔙 Go back")

    pick = questionary.select("🚀 Select method to run:", choices=display_choices).ask()
    return None if pick == "🔙 Go back" else pick.split()[1]


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
                print(f"▶️ Starting fine-tuning with {selected_method}...")
                subprocess.run(["bash", "./run_llmem.sh", model, str(batch_size), str(seq_length), *selected_method.split('+'), "true"])
                break


if __name__ == "__main__":
    main()