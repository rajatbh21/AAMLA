#!/usr/bin/env python3
import subprocess
import questionary
import sys
import torch
import pandas as pd
import argparse
import os
from transformers import AutoTokenizer, AutoModelForCausalLM

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
    banner = f.renderText('AAMLA')
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


def run_pass_at_k():
    while True:
        model_path = questionary.text("📁 Enter your fine-tuned model path:").ask().strip()
        if model_path and os.path.exists(model_path):
            break
        print(f"❌ Path not found: {model_path!r}")
        retry = questionary.select("🔁 Try again?", choices=["yes", "no"]).ask()
        if retry != "yes":
            print("⚠️ Skipping inference because model path is invalid.")
            return

    while True:
        output_path = questionary.text("📁 Enter output path:").ask().strip()
        if output_path and os.path.exists(output_path):
            break
        print(f"❌ Path not found: {output_path!r}")
    
    output_file = questionary.text("📁 Enter output file name:").ask().strip()

    num_infer = questionary.text("📁 Enter the number of inference per a task:").ask().strip()


    subprocess.run([
        "python", "model_inference/inference_VerilogEval.py",
        "--model", model_path,
        "--n", num_infer,
        "--temperature", "1.0",
        "--gpu_name", "0",
        "--output_dir", output_path,
        "--output_file", output_file + ".jsonl",
        "--bench_type", "Machine",
    ])

    # .jsonl to .v
    print("🚀 Transforming .jsonl to .v")
    subprocess.run(["python", "test_on_benchmark/jsonl2v.py",
                    "--input_jsonl", output_path + '/' + output_file + ".jsonl",
                    "--output_dir", output_path
                    ])
    
    # measure pass@k
    print("🚀 Running inference and measuring pass@k...")
    subprocess.run(["bash", "test_on_benchmark/run.sh",
                    "-p", output_path,
                    "-n", num_infer
                    ])


def run_chat_inference():
    model_id = questionary.text("🧠 Base / merged model path (HF repo id or local path):").ask()
    if not model_id:
        print("❌ model path is required.")
        return

    prompt = questionary.text("💬 Your prompt:").ask()
    if not prompt:
        print("❌ Empty prompt.")
        return

    temperature = questionary.text("🌡️ temperature (default 0.7):").ask() or "0.7"
    max_new_tokens = questionary.text("📝 max_new_tokens (default 256):").ask() or "256"
    gpu = questionary.text("🎯 GPU index — empty for auto:").ask().strip()

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    if gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
        device_map = "auto"
    else:
        device_map = "auto"

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    outputs = model.generate(
        **inputs,
        max_new_tokens=int(max_new_tokens),
        do_sample=True,
        temperature=float(temperature),
        top_p=0.95,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id is not None else None,
    )

    gen_only = outputs[0][inputs["input_ids"].shape[-1]:]
    text = tokenizer.decode(gen_only, skip_special_tokens=True)
    print("\n=== Response ===\n" + text + "\n")


def main():
    print_banner()

    parser = argparse.ArgumentParser()
    parser.add_argument("-acc", action="store_true")
    parser.add_argument("-time", action="store_true")
    args = parser.parse_args()

    if args.acc or args.time:
        model, batch_size, seq_length = get_inputs()
        estimates = get_memory_estimates(model, batch_size, seq_length)
        priority = "accuracy" if args.acc else "time"
        
        selected_method = select_method(estimates, priority)
        if selected_method:
            method_display = key_to_method_name[selected_method]
            print(f"▶️ Starting fine-tuning with {method_display}...")

            if selected_method == 'fft+lora':
                subprocess.run(["bash", "-c", "cd LLaMA-Factory && llamafactory-cli train examples/train_codellama/codellama_lora_sft.yaml"])
            elif selected_method == 'fft+dora':
                subprocess.run(["bash", "-c", "cd LLaMA-Factory && llamafactory-cli train examples/train_codellama/codellama_dora_sft.yaml"])
            elif selected_method == 'apollo+none':
                subprocess.run(["bash", "-c", "cd LLaMA-Factory && llamafactory-cli train examples/train_codellama/codellama_apollo_sft.yaml"])
            elif selected_method == 'mezo+none':
                subprocess.run(["bash", "-c", f"MODEL=codellama/CodeLlama-7b-Instruct-hf TASK=HaVen MODE=ft BS={batch_size} LR=1e-6 EPS=1e-4 bash MeZO/large_models/mezo.sh"])
            elif selected_method in ['tokentune+none', 'tokentune+lora']:
                subprocess.run(["bash", "tokentune/scripts/train/lora-tokentune-codellama.sh"])
        else:
            print("❌ No valid method fits in available GPU memory.")
        return  

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
                    step = 4

                elif selected_method == 'fft+dora':
                    subprocess.run(["bash", "-c", "cd LLaMA-Factory && llamafactory-cli train examples/train_codellama/codellama_dora_sft.yaml"])
                    step = 4

                elif selected_method == 'apollo+none':
                    subprocess.run(["bash", "-c", "cd LLaMA-Factory && llamafactory-cli train examples/train_codellama/codellama_apollo_sft.yaml"])
                    step = 4

                elif selected_method == 'mezo+none':
                    subprocess.run(["bash", "-c", "MODEL=codellama/CodeLlama-7b-Instruct-hf TASK=HaVen MODE=ft BS={} LR=1e-6 EPS=1e-4 bash MeZO/large_models/mezo.sh".format(batch_size)])
                    step = 4

                elif selected_method in ['tokentune+none', 'tokentune+lora']:
                    subprocess.run(["bash", "tokentune/scripts/train/lora-tokentune-codellama.sh"])
                    step = 4
        elif step == 4:
            while True:
                choice = questionary.select(
                    "Model Action:",
                    choices=[
                        "📈 Measure pass@k",
                        "💬 Chat inference",
                        "✅ Finish",
                    ],
                ).ask()

                if choice == "📈 Measure pass@k":
                    run_pass_at_k()
                elif choice == "💬 Chat inference":
                    run_chat_inference()
                elif choice == "✅ Finish" or choice is None:
                    print("✅ Pipeline finished.")
                    break

            break  




if __name__ == "__main__":
    main()