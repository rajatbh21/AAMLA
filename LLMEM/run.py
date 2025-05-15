import GPUtil
from pynvml import *
import torch.distributed as dist
from size_estimator import SizeEstimator
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoConfig
from peft import LoraConfig, get_peft_model, TaskType

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',   type=str, required=True, help="Model path on HuggingFace")
    parser.add_argument('--batch',   type=int, required=True, help="Batch size")
    parser.add_argument('--seq_len', type=int, required=True, help="Sequence length")

    # Separate method and PEFT
    parser.add_argument(
        '--method',
        choices=['none', 'mezo', 'tokentune'],
        default='none',
        help="Method to apply: 'none', 'mezo', or 'tokentune'"
    )

    parser.add_argument(
        '--peft',
        choices=['none', 'lora', 'dora'],
        default='none',
        help="PEFT method: 'none', 'lora', or 'dora'"
    )

    # PEFT-related arguments
    parser.add_argument('--lora_rank', type=int, default=8, help="LoRA rank")
    parser.add_argument(
        '--lora_target',
        nargs='+',
        default=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'up_proj', 'gate_proj', 'down_proj'],
        help="LoRA target modules"
    )

    parser.add_argument('--token_ratio', type=float, default=0.1, help="TokenTune token ratio")

    parser.add_argument('--gradient_checkpointing', type=bool, default=True, help="Enable gradient checkpointing")

    if not dist.is_initialized():
        dist.init_process_group(backend='nccl', init_method='env://')
    world_size = dist.get_world_size()
    
    args = parser.parse_args()
    model_name = args.model
    real_bs = args.batch
    seq_len = args.seq_len
    gradient_checkpointing = args.gradient_checkpointing

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(torch.cuda.get_device_properties(0).name)
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        torch_dtype=torch.bfloat16, # bf16
    ).to(device)

    if args.peft == 'lora':
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=args.lora_rank,
            lora_alpha=args.lora_rank * 2,
            lora_dropout=0.05,
            target_modules= [ 'q_proj', 'k_proj', 'v_proj', 'o_proj', 'up_proj', 'gate_proj', 'down_proj' ]
        )
        model = get_peft_model(model, lora_config)
        print(f"LoRA enabled: rank={args.lora_rank}, target={args.lora_target}")

    if args.peft == 'dora':
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=args.lora_rank,
            lora_alpha=args.lora_rank * 2,
            lora_dropout=0.05,
            target_modules= [ 'q_proj', 'k_proj', 'v_proj', 'o_proj', 'up_proj', 'gate_proj', 'down_proj' ],
            use_dora=True
        )
        model = get_peft_model(model, lora_config)
        print(f"DoRA enabled: rank={args.lora_rank}, target={args.lora_target}")

    print("Running LLMEM")

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    total_nvml = torch.cuda.get_device_properties(0).total_memory // (1024**2)
    reserved = torch.cuda.memory_reserved() // (1024**2)
    allocated = torch.cuda.memory_allocated() // (1024**2)
    cuda_context_mem = reserved - allocated

    def measure_cuda_context_overhead():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        initial = torch.cuda.memory_allocated() // (1024**2)
                    
        dummy = torch.ones(1).cuda()
        torch.cuda.synchronize()
                    
        context_overhead = torch.cuda.memory_allocated() // (1024**2) - initial
        return context_overhead

    context_overhead = measure_cuda_context_overhead()
    cuda_context_mem = max(cuda_context_mem, context_overhead)

    print(f"[0] Total GPU mem (MB): {total_nvml}")
    print(f"[0] Reserved/Allocated diff (MB): {cuda_context_mem}")

    vocab_size = model.config.vocab_size
    model.config.use_cache = False

    dummy_input_ids = torch.randint(
        low=0,
        high=vocab_size,
        size=(real_bs, seq_len),
        dtype=torch.long,
        device='cuda'
    )    
    batch = {
    "input_ids": dummy_input_ids
    }

    model_dtype_bytes = next(model.parameters()).element_size()
    input_dtype_bytes = batch[next(iter(batch))].element_size()

    batch_ids = batch["input_ids"][:real_bs]

    se = SizeEstimator(
        model=model,
        batch=batch_ids,
        real_bs=real_bs,
        bytes=model_dtype_bytes,
        bytes_input=input_dtype_bytes,
        gpu_n=world_size,
        tp=0,
        lm_fp32=True,
        m_total=total_nvml - cuda_context_mem,
        method=args.method,
        peft=args.peft,
        gradient_checkpointing=gradient_checkpointing,
        token_ratio=args.token_ratio
    )

    torch.cuda.empty_cache()
    prev_used = torch.cuda.memory_allocated() // (1024**2)
    model.gradient_checkpointing_disable()
    with torch.no_grad():
        se.get_output_sizes()
    model.gradient_checkpointing_enable()
    torch.cuda.empty_cache()
    after_used = torch.cuda.memory_allocated() // (1024**2)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model.eval()
    with torch.no_grad():
        _ = model(**batch, use_cache=False)

    peak = torch.cuda.max_memory_allocated() // (1024**2)
    extra_peak = max(peak - after_used, 0)
    print("Extra Peak: ", extra_peak)
    chunk_mem = torch.cuda.memory_allocated() // (1024**2)
    m_pbase = chunk_mem + cuda_context_mem - (after_used - prev_used)
    print(f"[m_pbase]: {m_pbase} MB")

    esti_mem, real_bs = se.estimate_size(m_init=m_pbase + extra_peak)
    print(f"Estimated peak memory: {esti_mem:.1f} MB with batch size {real_bs}")

if __name__== "__main__":
    main()