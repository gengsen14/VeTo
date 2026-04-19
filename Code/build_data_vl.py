import os
import json
import torch
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor, Qwen3VLForConditionalGeneration
import torch.multiprocessing as mp
import math
import ast
from PIL import Image
import io

GPU_IDS = [0, 1, 2, 3, 4, 5]

MODEL_PATH = "/mnt/models/Qwen3-VL-8B"
INPUT_FILE = "mmmu_math.parquet"
OUTPUT_FILE_BASE = "./parts/mmmu_8B_N16_logprobs"
FINAL_OUTPUT_FILE = "mmmu_math_qwen_vl_8B_N16_with_probs_merged.jsonl"

N_SAMPLES = 64
MICRO_BATCH_SIZE = 1
MAX_NEW_TOKENS = 2048
TEMPERATURE = 0.7
TOP_P = 0.95
TOP_K = 20
SAMPLE_SIZE = 250

def extract_answer(text, task_type="GPQ"):
    """Extract multiple-choice answers (A, B, C, D) in GPQ mode"""
    if not text:
        return None
    text = str(text).strip()
    candidate_str = None

    # Extract from \boxed{} first
    if "\\boxed" in text:
        idx = text.rfind("\\boxed")
        if idx != -1:
            open_brace_idx = text.find("{", idx)
            if open_brace_idx != -1:
                balance = 1
                for i in range(open_brace_idx + 1, len(text)):
                    char = text[i]
                    if char == '{':
                        balance += 1
                    elif char == '}':
                        balance -= 1

                    if balance == 0:
                        candidate_str = text[open_brace_idx + 1: i].strip()
                        break

    if not candidate_str:
        if "####" in text:
            parts = text.split("####")
            candidate_str = parts[-1].strip().split('\n')[0]
        else:
            match = re.search(r"(?:The answer is|Final Answer[:\s]*|The correct option is[:\s]*|Answer[:\s]*)\s*([^\n]+)", text, re.IGNORECASE)
            if match:
                candidate_str = match.group(1).strip()

    target_text = candidate_str if candidate_str else text

    # Match answer in brackets
    match_bracket = re.search(r"[\(\[\{]([A-Z])[\)\]\}]", target_text, re.IGNORECASE)
    if match_bracket:
        return match_bracket.group(1).upper()

    # Match answer at the start of string
    match_start = re.search(r"^(?:Option|Answer|Choice)?\s*([A-Z])(?:\.|:|\)|\s|$)", target_text, re.IGNORECASE)
    if match_start:
        return match_start.group(1).upper()

    # Clean non-alphabet characters
    clean_cand = re.sub(r"[^A-Z]", "", target_text.upper())
    if len(clean_cand) == 1:
        return clean_cand

    return None

def normalize_answer(ans):
    """Normalize answer string for comparison"""
    if not ans:
        return ""
    ans = str(ans).strip().replace("$", "").replace(" ", "").upper()
    return ans

def check_correctness(model_ans, gold_ans):
    """Check if model prediction matches ground truth"""
    return normalize_answer(model_ans) == normalize_answer(gold_ans)

def gpu_worker(rank, gpu_id, subset_data):
    print(f"Worker {rank} (GPU {gpu_id}) started with {len(subset_data)} samples...")

    try:
        # Use AutoProcessor for multi-modal tasks
        processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map={"": gpu_id}
        )
        model.eval()
        processor.tokenizer.padding_side = "left"
        print(f"Worker {rank} loaded model successfully (GPU: {gpu_id})")
    except Exception as e:
        print(f"Worker {rank} failed to load model: {e}")
        return

    out_file = f"{OUTPUT_FILE_BASE}_{rank}.jsonl"

    with open(out_file, 'w', encoding='utf-8') as f:
        pass

    for item in tqdm(subset_data, position=rank, desc=f"GPU {gpu_id}"):
        img_obj = None
        img_data = item.get("image_1")
        if img_data is not None:
            if isinstance(img_data, dict) and 'bytes' in img_data:
                img_obj = Image.open(io.BytesIO(img_data['bytes'])).convert('RGB')
            elif isinstance(img_data, Image.Image):
                img_obj = img_data.convert('RGB')

        raw_question = item.get("question", "")
        gold_answer = str(item.get("answer", ""))
        q_id = item.get("id", str(hash(raw_question)))

        # Parse and format options
        options = item.get("options", "[]")
        if isinstance(options, str):
            try:
                options = ast.literal_eval(options)
            except:
                pass

        options_text = ""
        if isinstance(options, list):
            for i, opt in enumerate(options):
                letter = chr(ord('A') + i)
                options_text += f"{letter}. {opt}\n"

        question = f"{raw_question}\n\nOptions:\n{options_text}" if options_text else raw_question
        instruction = "Please put the final answer option (e.g., A, B, C, D) within \\boxed{}."
        final_content = f"{question}\n\n{instruction}"

        # Build multi-modal input for Qwen-VL
        content_list = []
        if img_obj is not None:
            content_list.append({"type": "image", "image": img_obj})
        content_list.append({"type": "text", "text": final_content})

        messages = [{"role": "user", "content": content_list}]

        try:
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            ).to(f"cuda:{gpu_id}")
        except Exception as e:
            print(f"Tokenization error: {e}")
            continue

        prompt_len = inputs.input_ids.shape[1]
        candidates = []
        correct_count = 0

        # OOM protection with retry and batch size reduction
        max_retries = 5
        retry_count = 0
        current_micro_bs = MICRO_BATCH_SIZE

        while len(candidates) < N_SAMPLES and retry_count < max_retries:
            needed = N_SAMPLES - len(candidates)
            current_batch_size = min(current_micro_bs, needed)

            try:
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=MAX_NEW_TOKENS,
                        num_return_sequences=current_batch_size,
                        do_sample=True,
                        temperature=TEMPERATURE,
                        top_p=TOP_P,
                        repetition_penalty=1.0,
                        pad_token_id=processor.tokenizer.eos_token_id,
                        return_dict_in_generate=True,
                        output_scores=True
                    )

                    transition_scores = model.compute_transition_scores(
                        outputs.sequences, outputs.scores, normalize_logits=True
                    )

                    generated_sequences = outputs.sequences[:, prompt_len:]

                    for j in range(current_batch_size):
                        text = processor.tokenizer.decode(generated_sequences[j], skip_special_tokens=True)
                        extracted = extract_answer(text, task_type="GPQ")
                        is_correct = check_correctness(extracted, gold_answer)

                        logprobs = transition_scores[j].float().cpu().numpy().tolist()
                        token_ids = generated_sequences[j].cpu().tolist()

                        # Truncate logprobs at EOS token
                        if processor.tokenizer.eos_token_id in token_ids:
                            eos_idx = token_ids.index(processor.tokenizer.eos_token_id)
                            logprobs = logprobs[:eos_idx+1]

                        logprobs = [round(x, 4) for x in logprobs]

                        if is_correct:
                            correct_count += 1

                        candidates.append({
                            "text": text,
                            "extracted_answer": extracted,
                            "is_correct": is_correct,
                            "token_logprobs": logprobs
                        })

                    del outputs
                    del transition_scores
                    torch.cuda.empty_cache()
                    retry_count = 0

            except RuntimeError as e:
                torch.cuda.empty_cache()
                if "out of memory" in str(e).lower():
                    print(f"OOM warning: Worker {rank} collected {len(candidates)}/{N_SAMPLES}. Reducing to batch size 1...")
                    if current_micro_bs > 1:
                        current_micro_bs = 1
                    else:
                        retry_count += 1
                else:
                    print(f"Runtime error: {e}")
                    retry_count += 1

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

    print(f"Worker {rank} completed processing.")

def main():
    print(f"Reading input data: {INPUT_FILE}")
    if os.path.exists(INPUT_FILE):
        try:
            df = pd.read_parquet(INPUT_FILE, engine='pyarrow')

            if 'question_type' in df.columns:
                df = df[df['question_type'] == 'multiple-choice']
                print(f"Filtered multiple-choice questions: {len(df)}")

            if SAMPLE_SIZE > 0 and len(df) > SAMPLE_SIZE:
                print(f"Sampling {SAMPLE_SIZE} examples...")
                sampled_df = df.sample(n=SAMPLE_SIZE, random_state=42)
            else:
                sampled_df = df
            data = sampled_df.to_dict('records')
        except Exception as e:
            print(f"Error reading parquet file: {e}")
            return
    else:
        print("Input file not found.")
        return

    print(f"Total samples to process: {len(data)}")

    num_gpus = len(GPU_IDS)
    chunk_size = math.ceil(len(data) / num_gpus)

    mp.set_start_method('spawn', force=True)
    processes = []

    # Distribute data to GPU workers
    for rank, gpu_id in enumerate(GPU_IDS):
        start = rank * chunk_size
        end = min((rank + 1) * chunk_size, len(data))
        subset = data[start:end]

        p = mp.Process(target=gpu_worker, args=(rank, gpu_id, subset))
        p.start()
        processes.append(p)

    # Wait for all workers to finish
    for p in processes:
        p.join()

    print("All workers finished. Merging output files...")
    # Merge partial results into final file
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
