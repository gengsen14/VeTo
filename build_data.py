import os
import json
import random
import torch
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.multiprocessing as mp
import math
import gc

# Set this to the list of GPU IDs you want to use
GPU_IDS = [0, 1, 2, 3, 4, 5, 6, 7] 

MODEL_PATH = "/mnt/models/Qwen3-8B"
INPUT_FILE = "gsm8k.parquet"
# Intermediate Result
OUTPUT_FILE_BASE = "./parts/gsm8k_N16_logprobs"
# Final Result
FINAL_OUTPUT_FILE = "gsm8k_N16_with_probs_merged.jsonl"

# Sampling Config
N_SAMPLES = 16
MICRO_BATCH_SIZE = 1
MAX_NEW_TOKENS = 8192
TEMPERATURE = 0.7
TOP_P = 0.95
TOP_K = 20
# Set to -1 to process all
SAMPLE_SIZE = 250

def extract_answer(text, task_type="GPQ"):
    if not text: return None
    text = str(text).strip()
    candidate_str = None

    # Step 1: Candidate Extraction 
    # 1. Extract \boxed{} first
    if "\\boxed" in text:
        idx = text.rfind("\\boxed")
        if idx != -1:
            # Find the first '{' after \boxed
            open_brace_idx = text.find("{", idx)
            if open_brace_idx != -1:
                balance = 1
                for i in range(open_brace_idx + 1, len(text)):
                    char = text[i]
                    if char == '{': balance += 1
                    elif char == '}': balance -= 1
                    
                    if balance == 0:
                        candidate_str = text[open_brace_idx + 1 : i].strip()
                        break

    # 2. Fallback: If no \boxed found, find #### or text guide words
    if not candidate_str:
        if "####" in text:
            parts = text.split("####")
            candidate_str = parts[-1].strip().split('\n')[0]
        else:
            # Relax matching rules to capture the full sentence
            match = re.search(r"(?:The answer is|Final Answer[:\s]*|The correct option is[:\s]*|Answer[:\s]*)\s*([^\n]+)", text, re.IGNORECASE)
            if match: 
                candidate_str = match.group(1).strip()

    # Use full text as candidate if no extraction
    target_text = candidate_str if candidate_str else text
    
    # Step 2: Purify based on task type    
    if task_type == "GPQ":        
        # 1. Strict match: Options with brackets (A) [B] {C}
        match_bracket = re.search(r"[\(\[\{]([A-Z])[\)\]\}]", target_text, re.IGNORECASE)
        if match_bracket: 
            return match_bracket.group(1).upper()
            
        # 2. Start match: Option A, Answer: B, A., B:, C)
        match_start = re.search(r"^(?:Option|Answer|Choice)?\s*([A-Z])(?:\.|:|\)|\s|$)", target_text, re.IGNORECASE)
        if match_start:
            return match_start.group(1).upper()
            
        # 3. Pure letter fallback
        clean_cand = re.sub(r"[^A-Z]", "", target_text.upper())
        if len(clean_cand) == 1:
            return clean_cand
            
        return None

    elif task_type == "MATH":
        clean_text = target_text.replace(",", "").replace("$", "").replace("\\", "").replace("%", "")

        matches = re.findall(r"[-+]?\d*\.\d+|\d+", clean_text)
        if matches: return matches[-1]
        
        return None

    return None

def normalize_answer(ans):
    if not ans: return ""
    ans = str(ans).strip().replace("$", "").replace(" ", "")
    try:
        if "." in ans: return str(float(ans))
        return str(int(ans))
    except:
        return ans

def check_correctness(model_ans, gold_ans):
    return normalize_answer(model_ans) == normalize_answer(gold_ans)

def gpu_worker(rank, gpu_id, subset_data):
    """
    Worker process: Loads model on a specific GPU and processes a subset of data.
    """
    print(f"Worker {rank} (GPU {gpu_id}) started with {len(subset_data)} samples...")
    
    # Load Model on specific GPU
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, 
            torch_dtype=torch.bfloat16, 
            trust_remote_code=True,
            device_map=None 
        ).to(f"cuda:{gpu_id}")
        model.eval()
        tokenizer.padding_side = "left"
    except Exception as e:
        print(f"Worker {rank} Model Load Error: {e}")
        return

    out_file = f"{OUTPUT_FILE_BASE}_{rank}.jsonl"
    with open(out_file, 'w', encoding='utf-8') as f: pass

    for item in tqdm(subset_data, position=rank, desc=f"GPU {gpu_id}"):

        raw_question = item.get("Problem", item.get("problem", item.get("question", ""))) or item.get("prompt", "")

        # 1. Extract answer and ID
        gold_answer = str(item.get("Answer", item.get("answer", item.get("answerKey", ""))))
        q_id = item.get("question_id", item.get("id", str(hash(raw_question))))

        # 2. Process multiple-choice questions
        options = item.get("options", item.get("choices", []))
        if options and len(options) > 0:
            options_text = []
            for i in range(len(options.get("label", []))):
                letter = options.get("label")[i]
                opt = options.get("text")[i]
                options_text.append(f"{letter}. {opt}")
            
            question = f"Question: {raw_question}\nOptions:\n" + "\n".join(options_text)
        else:
            question = raw_question

        # 3. Dynamically assemble instruction
        instruction = "Please put the final answer within \\boxed{}."

        # 4. Final concatenation
        final_content = f"{question}\n\n{instruction}"

        messages = [{"role": "user", "content": final_content}]
        try:
            inputs = tokenizer.apply_chat_template(
                messages, 
                return_tensors="pt", 
                return_dict=True,
                add_generation_prompt=True, 
                enable_thinking=True
            ).to(f"cuda:{gpu_id}")
        except Exception as e:
            print(f"Error tokenizing: {e}")
            continue
            
        prompt_len = inputs.input_ids.shape[1]
        candidates = []
        correct_count = 0
        
        # Calculate loops needed
        num_loops = (N_SAMPLES + MICRO_BATCH_SIZE - 1) // MICRO_BATCH_SIZE
        
        for loop_idx in range(num_loops):
            current_batch_size = min(MICRO_BATCH_SIZE, N_SAMPLES - loop_idx * MICRO_BATCH_SIZE)
            
            try:
                with torch.no_grad():
                    # Generate
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=MAX_NEW_TOKENS,
                        num_return_sequences=current_batch_size,
                        do_sample=True,
                        temperature=TEMPERATURE,
                        top_p=TOP_P,
                        repetition_penalty=1.0,
                        pad_token_id=tokenizer.eos_token_id,
                        return_dict_in_generate=True,
                        output_scores=True 
                    )
                    
                    # Compute Logprobs
                    transition_scores = model.compute_transition_scores(
                        outputs.sequences, outputs.scores, normalize_logits=True
                    )
                    
                    generated_sequences = outputs.sequences[:, prompt_len:]
                    
                    for j in range(current_batch_size):
                        text = tokenizer.decode(generated_sequences[j], skip_special_tokens=True)
                        extracted = extract_answer(text)
                        is_correct = check_correctness(extracted, gold_answer)
                        
                        # Extract logprobs ignoring padding/special tokens if needed
                        logprobs = transition_scores[j].float().cpu().numpy().tolist()
                        
                        token_ids = generated_sequences[j].cpu().tolist()
                        if tokenizer.eos_token_id in token_ids:
                            eos_idx = token_ids.index(tokenizer.eos_token_id)
                            logprobs = logprobs[:eos_idx+1]
                        
                        logprobs = [round(x, 4) for x in logprobs]
                        
                        if is_correct: correct_count += 1
                        
                        candidates.append({
                            "text": text,
                            "extracted_answer": extracted,
                            "is_correct": is_correct,
                            "token_logprobs": logprobs
                        })
                    
                    del outputs
                    del transition_scores
                    torch.cuda.empty_cache()
                    
            except RuntimeError as e:
                print(f"OOM/Error on Worker {rank} (Loop {loop_idx}): {e}")
                torch.cuda.empty_cache()
                continue 

        # Save record
        record = {
            "question_id": q_id,
            "problem": question,
            "gold_answer": gold_answer,
            "candidates": candidates,
            "stats": {
                "total": len(candidates),
                "correct": correct_count,
                "accuracy": correct_count / len(candidates) if candidates else 0
            }
        }        
        with open(out_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Worker {rank} Finished.")


def main():
    # 1. Load dataset
    print(f"Reading Data: {INPUT_FILE}")
    if os.path.exists(INPUT_FILE):
        try:
            df = pd.read_parquet(INPUT_FILE, engine='pyarrow')
            if SAMPLE_SIZE > 0 and len(df) > SAMPLE_SIZE:
                print(f"Sampling {SAMPLE_SIZE} examples...")
                sampled_df = df.sample(n=SAMPLE_SIZE, random_state=42)
            else:
                sampled_df = df
            data = sampled_df.to_dict('records')
        except Exception as e:
            print(f"Error reading parquet: {e}")
            return
    else:
        print("Input file not found.")
        return

    print(f"Total Problems: {len(data)}")
    
    # 2. Split data for GPUs
    num_gpus = len(GPU_IDS)
    chunk_size = math.ceil(len(data) / num_gpus)
    
    # 3. Start processes
    mp.set_start_method('spawn', force=True)
    processes = []
    
    for rank, gpu_id in enumerate(GPU_IDS):
        start = rank * chunk_size
        end = min((rank + 1) * chunk_size, len(data))
        subset = data[start:end]
        
        p = mp.Process(target=gpu_worker, args=(rank, gpu_id, subset))
        p.start()
        processes.append(p)
    
    # 4. Wait for all processes to finish
    for p in processes:
        p.join()
        
    # 5. Merge all result files
    print("All workers finished. Merging files...")
    with open(FINAL_OUTPUT_FILE, 'w', encoding='utf-8') as outfile:
        for rank in range(num_gpus):
            part_file = f"{OUTPUT_FILE_BASE}_{rank}.jsonl"
            if os.path.exists(part_file):
                with open(part_file, 'r', encoding='utf-8') as infile:
                    for line in infile:
                        outfile.write(line)
    
    print(f"Final merged file saved to: {FINAL_OUTPUT_FILE}")

if __name__ == "__main__":
    main()
