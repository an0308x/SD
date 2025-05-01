import torch
import argparse
import contexttimer
from colorama import Fore, Style
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from sampling import autoregressive_sampling, speculative_sampling, speculative_sampling_v2
from globals import Decoder

# my local models
MODELZOO = {
    # llama-1
    # https://huggingface.co/PY007/TinyLlama-1.1B-step-50K-105b
    "llama1b": "/share_nfs/fangjiarui/root/code/hf_models/TinyLlama-1.1B-step-50K-105b",
    "llama7b": "/share_nfs/tianzhi/code/llama-7b",
    "llama30b": "/share_nfs/fangjiarui/root/code/hf_models/llama-30b-hf",
    "llama2-7b" : "/share_nfs/fangjiarui/root/code/hf_models/llama-2-7b-hf",
    "llama2-70b" : "/share_nfs/fangjiarui/root/code/hf_models/llama-2-70b-hf",
    "bloom-560m": "/share_nfs/fangjiarui/root/code/hf_models/bloom-560m",
    "bloom7b": "/share_nfs/fangjiarui/root/code/hf_models/bloomz-7b1",
    "baichuan-7b": "/share_nfs/duanqiyuan/models/source_models/hf/baichuan-7B",
    "baichuan-13b": "/share_nfs/duanqiyuan/models/source_models/hf/Baichuan-13B-Base",
}

def parse_arguments():
    parser = argparse.ArgumentParser(description='args for main.py')

    parser.add_argument('--input', type=str, default="Any recommendations for my holidays in Abu Dhabi?")
    parser.add_argument('--approx_model_name', type=str, default=MODELZOO["llama2-7b"])
    parser.add_argument('--target_model_name', type=str, default=MODELZOO["llama2-70b"])
    parser.add_argument('--verbose', '-v', action='store_true', default=False, help='enable verbose mode')
    parser.add_argument('--seed', '-s', type=int, default=None, help='set a random seed, which can makes the result reproducible')
    parser.add_argument('--benchmark', '-b', action='store_true', default=False, help='show benchmark results.')
    parser.add_argument('--profiling', '-p', action='store_true', default=False, help='collect torch profiler results.')
    parser.add_argument('--max_tokens', '-M', type=int, default=20, help='max token number generated.')
    parser.add_argument('--gamma', '-g', type=int, default=4, help='guess time.')
    args = parser.parse_args()
    return args


def color_print(text):
    print(Fore.RED + text + Style.RESET_ALL)
    
def benchmark(fn, print_prefix, use_profiler=True, *args, **kwargs):
    TEST_TIME = 10
    profile_filename = f"./profile_logs/{print_prefix}"
    
    with contexttimer.Timer() as t:
        if use_profiler:
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(wait=0, warmup=1, active=2, repeat=1, skip_first=0),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(profile_filename),
                record_shapes=False,
                profile_memory=False,
                # with_stack=True
            ) as prof:
                for _ in range(TEST_TIME): 
                    output = fn(*args, **kwargs)
                    prof.step()
        else:
            for _ in range(TEST_TIME): 
                output = fn(*args, **kwargs)

    print(f"\n [benchmark] {print_prefix}, tokens/sec: {len(output[0]) / t.elapsed / TEST_TIME}, {t.elapsed / TEST_TIME} sec generates {len(output[0])} tokens")

def generate(input_text, approx_model_name, target_model_name, num_tokens=20, gamma = 4,
             random_seed = None, verbose = False, use_benchmark = False, use_profiling = False):
    # NOTE() approx_model_name and target_model_name should use the same tokenizer!
    
    torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    tokenizer = AutoTokenizer.from_pretrained(approx_model_name, trust_remote_code=True)
  
    Decoder().set_tokenizer(tokenizer)
    
    print(f"begin loading models: \n {approx_model_name} \n {target_model_name}")

    # Load approx model with config adjustment
    print(f"Loading config for approx model: {approx_model_name}")
    try:
        approx_config = AutoConfig.from_pretrained(approx_model_name, trust_remote_code=True)
        if hasattr(approx_config, "rope_scaling") and isinstance(approx_config.rope_scaling, dict):
            print(f"Original approx rope_scaling: {approx_config.rope_scaling}")
            required_keys = {'type', 'factor'}
            if set(approx_config.rope_scaling.keys()) != required_keys:
                original_rope_config = approx_config.rope_scaling
                rope_type = original_rope_config.get('rope_type')
                factor = original_rope_config.get('factor')
                if rope_type is not None and factor is not None:
                     new_rope_scaling = {'type': rope_type, 'factor': factor}
                     print(f"Modifying approx rope_scaling to: {new_rope_scaling}")
                     approx_config.rope_scaling = new_rope_scaling
                else:
                     print(f"Warning: Could not find 'rope_type' or 'factor' in approx rope_scaling config: {original_rope_config}, attempting to load without modification.")
        
        print("Loading approx model...")
        small_model = AutoModelForCausalLM.from_pretrained(approx_model_name,
                                                         config=approx_config,
                                                         trust_remote_code=True,
                                                         # torch_dtype=torch.float16
                                                        ).to(torch_device)
        print("Approx model loaded.")
    except Exception as e:
        print(f"Error loading approx model {approx_model_name}: {e}")
        raise

    # Load target model with config adjustment
    print(f"Loading config for target model: {target_model_name}")
    try:
        target_config = AutoConfig.from_pretrained(target_model_name, trust_remote_code=True)
        if hasattr(target_config, "rope_scaling") and isinstance(target_config.rope_scaling, dict):
            print(f"Original target rope_scaling: {target_config.rope_scaling}")
            required_keys = {'type', 'factor'}
            if set(target_config.rope_scaling.keys()) != required_keys:
                original_rope_config = target_config.rope_scaling
                rope_type = original_rope_config.get('rope_type')
                factor = original_rope_config.get('factor')
                if rope_type is not None and factor is not None:
                    new_rope_scaling = {'type': rope_type, 'factor': factor}
                    print(f"Modifying target rope_scaling to: {new_rope_scaling}")
                    target_config.rope_scaling = new_rope_scaling
                else:
                    print(f"Warning: Could not find 'rope_type' or 'factor' in target rope_scaling config: {original_rope_config}, attempting to load without modification.")

        print("Loading target model...")
        large_model = AutoModelForCausalLM.from_pretrained(target_model_name,
                                                         config=target_config,
                                                         trust_remote_code=True,
                                                         # torch_dtype=torch.bfloat16
                                                        ).to(torch_device)
        print("Target model loaded.")
    except Exception as e:
        print(f"Error loading target model {target_model_name}: {e}")
        raise

    print("finish loading models")

    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(torch_device)

    top_k = 20
    top_p = 0.9

    torch.manual_seed(123)
    output = autoregressive_sampling(input_ids, large_model, num_tokens, top_k = top_k, top_p=top_p)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    color_print(f"large (target) model autoregressive_sampling: {generated_text}")
    
    if use_benchmark:
        benchmark(autoregressive_sampling, "AS_large", use_profiling,
                  input_ids, large_model, num_tokens, top_k = top_k, top_p=top_p)

    torch.manual_seed(123)
    output = autoregressive_sampling(input_ids, small_model, num_tokens, top_k = top_k, top_p=top_p)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    color_print(f"small (approx) model autoregressive_sampling: {generated_text}")
    
    if use_benchmark:
        benchmark(autoregressive_sampling, "AS_small", use_profiling,
                  input_ids, small_model, num_tokens, top_k = top_k, top_p=top_p)
    
    torch.manual_seed(123)
    output = speculative_sampling_v2(input_ids, small_model, large_model, num_tokens, top_k = top_k, top_p=top_p, random_seed = random_seed)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    color_print(f"deepmind's speculative_sampling: {generated_text}")   

    torch.manual_seed(123)
    output = speculative_sampling(input_ids, small_model, large_model, num_tokens, gamma = gamma, top_k = top_k, top_p=top_p, random_seed = random_seed, verbose = verbose)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    color_print(f"google's speculative_sampling: {generated_text}")
    
    if use_benchmark:
        benchmark(speculative_sampling, "SP", use_profiling,
                  input_ids, small_model, large_model, max_len = num_tokens, gamma = gamma, top_k = top_k, top_p=top_p, random_seed = random_seed)

if __name__ == "__main__":
    args = parse_arguments()
    
    generate(args.input, args.approx_model_name, args.target_model_name, num_tokens=args.max_tokens, gamma=args.gamma,
             random_seed = args.seed, verbose=args.verbose, use_benchmark = args.benchmark, use_profiling = args.profiling)
