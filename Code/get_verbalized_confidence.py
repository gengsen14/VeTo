import os
import json
import torch
import re
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.multiprocessing as mp

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
MODEL_PATH = "/mnt/models/Qwen3-8B"

INPUT_FILE = "rbench_8B_think_N16_with_probs_merged_FIXED.jsonl"
OUTPUT_FILE_BASE = "./parts/rbench_8B_think_N16_interval_part"
final_file = "./score/rbench_8B_think_N16_scored.jsonl"

is_poison = False

NUM_INTERVALS = 10 
EVAL_BATCH_SIZE = 8 
SPLIT_FLAG= '\n\n'

USE_MMLU_PROMPT = (
    "Task: Strict Reasoning Path Verification.\n"
    "Analyze the current thought process against the problem constraints. Logic must be sound and consistent.\n\n"
    "On a scale of 0-100, what is your confidence score regarding the eventual success of this path?\n"
    "**CRITERIA:**\n"
    "- **High Score**: The reasoning step represents a valid logical deduction that advances towards the solution.\n"
    "- **Low Score**: \n"
    "  1. Introduces numbers or assumptions without basis (Hallucinations).\n"
    "  2. The logic contradicts the problem statement or previous steps.\n"
    "  3. The reasoning method is fundamentally flawed or leads to a dead end.\n\n"
    "Note: Be harsh. If the thought process feels 'drifting' or 'inconsistent', give 0%.\n\n"
    "### Your Turn:\n"
    "Output strictly in this format: **Confidence: [Score]%**"
)

USE_MMLU_PROMPT_SIMPLE = (
    "Task: Strict Reasoning Path Verification.\n"
    "Analyze the current thought process against the problem constraints. Logic must be sound and consistent.\n\n"
    "**CRITERIA:**\n"
    "- **100%**: The reasoning step represents a valid logical deduction that advances towards the solution.\n"
    "- **0%**: \n"
    "  1. Introduces numbers or assumptions without basis (Hallucinations).\n"
    "  2. The logic contradicts the problem statement or previous steps.\n"
    "  3. The reasoning method is fundamentally flawed or leads to a dead end.\n\n"
    "Note: Be harsh. If the thought process feels 'drifting' or 'inconsistent', give 0%.\n\n"
    "### Your Turn:\n"
    "Output strictly in this format: **Confidence: [Score]%**"
)

USE_GSM8K_PROMPT = (
    "Task: Strict Step-by-Step Verification.\n"
    "Analyze the latest step against the problem description. There is NO room for ambiguity in Grade School Math.\n\n"
    "On a scale of 0-100, what is your confidence score regarding the eventual success of this path?\n"
    "**CRITERIA:**\n"
    "- **High Score**: The step clearly follows from the previous sentence and the numbers align perfectly with the problem.\n"
    "- **Low Score**: \n"
    "  1. Any number appears without a clear source (Magic Numbers).\n"
    "  2. The operation doesn't make sense in the real-world context.\n"
    "  3. The calculation is wrong.\n\n"
    "Note: Be harsh. If the logic feels 'jumpy' or 'weird', give 0%.\n\n"
    "### Your Turn:\n"
    "Output strictly in this format: **Confidence: [Score]%**"
)

USE_GSM8K_PROMPT_SIMPLE = (
    "Task: Strict Step-by-Step Verification.\n"
    "Analyze the latest step against the problem description. There is NO room for ambiguity in Grade School Math.\n\n"
    "**CRITERIA:**\n"
    "- **100%**: The step clearly follows from the previous sentence and the numbers align perfectly with the problem.\n"
    "- **0%**: \n"
    "  1. Any number appears without a clear source (Magic Numbers).\n"
    "  2. The operation doesn't make sense in the real-world context.\n"
    "  3. The calculation is wrong.\n\n"
    "Note: Be harsh. If the logic feels 'jumpy' or 'weird', give 0%.\n\n"
    "### Your Turn:\n"
    "Output strictly in this format: **Confidence: [Score]%**"
)

SCORE_PATTERN = re.compile(r"Confidence[:\s]*(\d+)", re.IGNORECASE)

def get_interval_slices(text, num_intervals=5, is_poison=False, poison_prefix=""):
    text = text.replace("<|im_end|>", "").strip()
    
    clean_prefix = ""
    generation = text
    
    # 1. Separate prefix and generated content
    if is_poison and poison_prefix and poison_prefix.strip():
        clean_prefix = poison_prefix.strip()
        if text.startswith(clean_prefix):
            generation = text[len(clean_prefix):].strip()

    # 2. Extract steps
    steps = [s for s in generation.split(SPLIT_FLAG) if s.strip()]
    if not steps:
        return [text]

    # 3. Calculate target length for base slices
    total_len = sum(len(s) for s in steps) + (len(steps) - 1) * len(SPLIT_FLAG)
    target_len = total_len / max(1, num_intervals)
    
    raw_slices = []
    current_buffer = []
    current_len = 0
    
    for step in steps:
        current_buffer.append(step)
        current_len += len(step) + len(SPLIT_FLAG)
        
        if current_len >= target_len and len(raw_slices) < num_intervals - 1:
            raw_slices.append(SPLIT_FLAG.join(current_buffer))
            current_buffer = []
            current_len = 0
            
    if current_buffer:
        raw_slices.append(SPLIT_FLAG.join(current_buffer))

    # 4. Construct cumulative slices
    cumulative_slices = []
    accumulated_text = []
    base_start = f"{clean_prefix}\n" if clean_prefix else ""
    
    for segment in raw_slices:
        accumulated_text.append(segment)
        cumulative_slices.append(base_start + SPLIT_FLAG.join(accumulated_text))
        
    if not cumulative_slices:
        return [text]
        
    # 5. Return based on current logic: Full text first, then the very first interval.
    if len(cumulative_slices) > 1:
        return [cumulative_slices[-1], cumulative_slices[0]]
    else:
        return [cumulative_slices[-1]]

def gpu_worker(rank, gpu_id, subset_data):
    """
    Worker process for streaming inferences to disk.
    """
    print(f"Worker {rank} (GPU {gpu_id}) started with {len(subset_data)} problems...")
    
    # Prepare output directory
    if not os.path.exists(os.path.dirname(OUTPUT_FILE_BASE)):
        os.makedirs(os.path.dirname(OUTPUT_FILE_BASE), exist_ok=True)
    out_file = f"{OUTPUT_FILE_BASE}_{rank}.jsonl"

    # Load Model
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, 
            torch_dtype=torch.bfloat16, 
            trust_remote_code=True,
            device_map=None 
        ).to(f"cuda:{gpu_id}")
        model.eval()
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
    except Exception as e:
        print(f"Worker {rank} Model Load Error: {e}")
        return

    with open(out_file, 'w', encoding='utf-8') as f_out:
        for record in tqdm(subset_data, position=rank, desc=f"GPU {gpu_id}"):
            try:
                question = record.get("problem", "")
                candidates = record.get("candidates", record.get("continuations", []))
                poison_prefix = record.get("full_poisoned_prefix", "")

                if not candidates:
                    f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f_out.flush() 
                    continue

                # Prepare Prompts
                flat_prompts = []
                metadata = [] 
                
                for cand_idx, cand in enumerate(candidates):
                    cand_text = cand.get("text", cand.get("full_text", ""))
                    
                    slices = get_interval_slices(
                        cand_text, 
                        num_intervals=NUM_INTERVALS, 
                        is_poison=is_poison, 
                        poison_prefix=poison_prefix
                    )
                    
                    for slice_idx, origin_slice_text in enumerate(slices):
                        slice_text = re.sub(r'<think>.*?</think>', '', origin_slice_text, flags=re.DOTALL).strip()
                        messages = [
                            {"role": "user", "content": question},
                            {"role": "assistant", "content": slice_text},
                            {"role": "user", "content": USE_MMLU_PROMPT}
                        ]
                        
                        text_input = tokenizer.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
                        )
                        flat_prompts.append(text_input)
                        metadata.append((cand_idx, slice_idx))

                # Batch Inference
                all_scores_flat = []
                num_prompts = len(flat_prompts)
                
                for i in range(0, num_prompts, EVAL_BATCH_SIZE):
                    sub_batch = flat_prompts[i : i + EVAL_BATCH_SIZE]
                    try:
                        inputs = tokenizer(
                            sub_batch, 
                            return_tensors="pt", 
                            padding=True, 
                            truncation=True, 
                            max_length=8192
                        ).to(f"cuda:{gpu_id}")
                        
                        with torch.no_grad():
                            outputs = model.generate(
                                **inputs,
                                max_new_tokens=1024*4,
                                do_sample=False, 
                                pad_token_id=tokenizer.eos_token_id
                            )
                        
                        input_len = inputs.input_ids.shape[1]
                        generated_ids = outputs[:, input_len:]
                        decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                        
                        for text in decoded:
                            ans_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
                            match = SCORE_PATTERN.search(ans_text)

                            if match:
                                all_scores_flat.append(min(100, max(0, int(match.group(1)))))
                            else:
                                all_scores_flat.append(0)
                                
                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            print(f"OOM on GPU {gpu_id}. Clearing cache.")
                            torch.cuda.empty_cache()
                            all_scores_flat.extend([0] * len(sub_batch))
                        else:
                            print(f"Inference Error on GPU {gpu_id}: {e}")
                            all_scores_flat.extend([0] * len(sub_batch))

                # Aggregate Results
                cand_score_map = {i: [] for i in range(len(candidates))}
                for idx, score in enumerate(all_scores_flat):
                    c_idx, _ = metadata[idx]
                    cand_score_map[c_idx].append(score)
                
                # Update Candidates
                for i, cand in enumerate(candidates):
                    raw_scores = cand_score_map[i]
                    if not raw_scores:
                        avg_s, min_s = 0, 0
                    else:
                        avg_s = sum(raw_scores) / len(raw_scores)
                        min_s = min(raw_scores)
                    
                    cand["interval_scores"] = raw_scores
                    cand["avg_verbal_score"] = avg_s
                    cand["min_verbal_score"] = min_s
                    cand["verbal_score"] = avg_s 

                # Calculate statistics
                if candidates:
                    record["stats"]["avg_confidence"] = sum(c["avg_verbal_score"] for c in candidates) / len(candidates)

                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                f_out.flush() 

            except Exception as inner_e:
                print(f"Error processing record in Worker {rank}: {inner_e}")
                continue

    print(f"Worker {rank} Done. Saved to {out_file}")

def main():
    print(f"Reading Data: {INPUT_FILE}")
    data = []
    if os.path.exists(INPUT_FILE):
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        data.append(json.loads(line))
                    except:
                        pass
    else:
        print("Input file not found.")
        return

    print(f"Total Problems: {len(data)}")
    
    num_gpus = torch.cuda.device_count()
    print(f"Detected {num_gpus} GPUs available.")
    
    chunk_size = math.ceil(len(data) / num_gpus)
    processes = []
    
    mp.set_start_method('spawn', force=True)

    for rank in range(num_gpus):
        start = rank * chunk_size
        end = min((rank + 1) * chunk_size, len(data))
        subset = data[start:end]
        
        if not subset:
            continue
            
        p = mp.Process(target=gpu_worker, args=(rank, rank, subset))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("All workers finished. Merging...")
    
    with open(final_file, 'w', encoding='utf-8') as outfile:
        count = 0
        for rank in range(num_gpus):
            part_file = f"{OUTPUT_FILE_BASE}_{rank}.jsonl"
            if os.path.exists(part_file):
                print(f"  -> Merging {part_file}...")
                with open(part_file, 'r', encoding='utf-8') as infile:
                    for line in infile:
                        outfile.write(line)
                        count += 1
            else:
                print(f"  Warning: {part_file} not found (Worker probably crashed early).")
    
    print(f"Final Dataset Saved: {final_file} (Total Lines: {count})")

if __name__ == "__main__":
    main()