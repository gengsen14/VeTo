import json
import numpy as np
import math
import os
import argparse 
from tqdm import tqdm
from collections import Counter

# Helper Functions
def calculate_peak_step_metrics(probs, block_size=10, i_start=0.25, i_end=0.75):
    p_arr = np.array(probs)
    n = len(p_arr)
    if n < block_size: return 0.0
    
    block_jitters = [np.mean(p_arr[i : i + block_size]) for i in range(0, n, block_size) if len(p_arr[i : i + block_size]) >= 2]
    if len(block_jitters) < 2: return 0.0
    
    diffs = np.abs(np.diff(block_jitters))
    if len(diffs) == 0: return 0.0
    
    sorted_diffs = np.sort(diffs)
    n_diff = len(sorted_diffs)
    cut1, cut2 = int(n_diff * i_start), int(n_diff * i_end)
    
    middle_diffs = sorted_diffs[cut1:cut2] if cut1 > 0 else sorted_diffs
    return np.mean(middle_diffs) if len(middle_diffs) > 0 else 0.0

def get_threshold_from_percentile(data, ratio, direction="top"):
    if ratio <= 0 or not data: return None
    ratio = max(0.0, min(100.0, ratio))
    return np.percentile(data, 100 - ratio) if direction == "top" else np.percentile(data, ratio)

def normalize_answer(ans):
    if ans is None: return ""
    ans = str(ans).strip().replace(" ", "")
    if ans.startswith("$"): ans = ans[1:]
    try:
        return str(float(ans)) if "." in ans else str(int(ans))
    except:
        return ans

def get_correct_count(record, candidates):
    count = 0
    gold = normalize_answer(record.get("gold_answer", ""))
    for c in candidates:
        if c.get("is_correct", False) or (gold and normalize_answer(c.get("extracted_answer", "")) == gold):
            count += 1
            c["is_correct"] = True
    return count

def execute_voting(record, pool):
    if not pool: return False, 0.0, [], ""
    cand_ans_pairs = []
    answers = []
    gold = normalize_answer(record.get("gold_answer", ""))
    
    for cand in pool:
        ans = normalize_answer(cand.get("extracted_answer", ""))
        if not ans:
            ans = gold if cand.get("is_correct", False) and gold else "INVALID"
        answers.append(ans)
        cand_ans_pairs.append((cand, ans))
    
    if not answers: return False, 0.0, [], ""
    
    counts = Counter(answers)
    best_ans, best_count = counts.most_common(1)[0]
    confidence = best_count / len(pool)
    best_pool = [cand for cand, ans in cand_ans_pairs if ans == best_ans]
    is_correct = (gold and best_ans == gold) or any(c.get("is_correct", False) for c in best_pool)
    
    return is_correct, confidence, best_pool, best_ans

WINDOW_SIZE = 20
interval_start = 0.25
interval_end = 0.75

def main(args):
    INPUT_FILE = args.input
    OUTPUT_FILE = args.output
    
    SAMPLE_S = args.sample_s
    SAMPLE_E = args.sample_e
    
    V100_JITTER_TOP_RATIO = args.v100_jitter
    V0_JITTER_KEEP_TOP_RATIO = args.v0_jitter
    MIN_SCORE = args.min_score

    print("=" * 60)
    print("RUNNING STRATEGY: [ JITTER ]")
    print(f"Input: {INPUT_FILE}")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Jitter Params: V100_J({V100_JITTER_TOP_RATIO}), V0_J({V0_JITTER_KEEP_TOP_RATIO})")
    print("=" * 60)
    
    if os.path.exists(OUTPUT_FILE): os.remove(OUTPUT_FILE)
    
    all_records = []
    stats_all_jitter = []
    
    print("Pass 1: Loading Data & Computing Features...")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                rec = json.loads(line)
                cands = rec.get("candidates", [])[SAMPLE_S:SAMPLE_E]
                if not cands: continue
                
                for c in cands:
                    c['_score'] = c.get("interval_scores", [0])[0] if c.get("interval_scores") else 0
                    
                    logs = c.get("token_logprobs", [])
                    probs = [math.exp(lp) for lp in logs if lp is not None and lp > -100]
                    
                    c['_jitter'] = calculate_peak_step_metrics(probs, WINDOW_SIZE, interval_start, interval_end)
                    stats_all_jitter.append(c['_jitter'])
                    
                all_records.append(rec)
            except Exception:
                continue

    total_problems = len(all_records)
    print(f"Loaded {total_problems} valid records.")

    # Calculate global Jitter thresholds
    t_v100_j = get_threshold_from_percentile(stats_all_jitter, V100_JITTER_TOP_RATIO, "top")
    t_v0_j   = get_threshold_from_percentile(stats_all_jitter, V0_JITTER_KEEP_TOP_RATIO, "top")
    
    thresh_v100_jitter = t_v100_j if t_v100_j is not None else 10000
    thresh_v0_keep     = t_v0_j if t_v0_j is not None else 10000

    stats_base, hits_base = [], 0
    stats_final, hits_final = [], 0

    with open(OUTPUT_FILE, "a", encoding="utf-8") as f_out:
        for rec in tqdm(all_records, desc="Executing Voting Strategy"):
            pool = rec['candidates'][SAMPLE_S:SAMPLE_E]
            
            # Extract all normalized answers from the original pool
            gold = normalize_answer(rec.get("gold_answer", ""))
            original_ans_list = []
            for cand in pool:
                ans = normalize_answer(cand.get("extracted_answer", ""))
                if not ans:
                    ans = gold if cand.get("is_correct", False) and gold else "INVALID"
                original_ans_list.append(ans)
            
            stats_base.append((get_correct_count(rec, pool), len(pool)))
            is_corr_base, conf_base, _, base_ans = execute_voting(rec, pool)
            if is_corr_base: hits_base += 1

            pool_s1 = [cand for cand in pool if cand['_score'] >= MIN_SCORE]
            if not pool_s1: pool_s1 = pool[:] 
            
            pool_s2 = []
            for cand in pool_s1:
                keep = True
                if cand['_score'] >= MIN_SCORE:
                    if cand['_jitter'] > thresh_v100_jitter: 
                        keep = False
                else:
                    if cand['_jitter'] < thresh_v0_keep: 
                        keep = False
                if keep: pool_s2.append(cand)
            
            if not pool_s2: pool_s2 = pool_s1[:]
            is_corr_final, conf_final, final_pool, final_ans = execute_voting(rec, pool_s2)

            # Calculate ratio of the final answer in the original pool
            final_ans_count = original_ans_list.count(final_ans)
            final_ans_original_ratio = final_ans_count / len(pool) if pool else 0.0

            stats_final.append((get_correct_count(rec, final_pool), len(final_pool)))
            if is_corr_final: hits_final += 1

            final_scores = [cand.get("_score", 0) for cand in final_pool]
            final_corrects = [cand.get("is_correct", False) for cand in final_pool]

            out = {
                "question_id": rec.get("question_id"),
                "is_correct": is_corr_final,
                "strategy": "JITTER",
                "is_same_as_base": (final_ans == base_ans), 
                "final_ans_original_ratio": final_ans_original_ratio,
                "pool_size": len(final_pool),
                "baseline_confidence": conf_base,
                "final_confidence": conf_final, 
                "total_tokens": 0, 
                "stop_reason": "final verbalize",
                "current_score": final_scores,
                "current_corrrect": final_corrects
            }
            f_out.write(json.dumps(out) + "\n")

    def calc_metrics(stats, hits):
        total_evals = len(stats)
        if total_evals == 0: return 0,0,0
        acc = (hits / total_evals) * 100
        oracle = np.mean([1 if c > 0 else 0 for c, n in stats]) * 100
        size = np.mean([n for c, n in stats])
        return acc, oracle, size

    m_base = calc_metrics(stats_base, hits_base)
    m_final = calc_metrics(stats_final, hits_final)

    print("\n" + "="*70)
    print(f"PERFORMANCE REPORT (N={len(stats_final)} paths computed)")
    print("="*70)
    print(f"{'Metric':<20} | {'Baseline':<18} | {'Final Strategy':<18}")
    print("-" * 70)
    
    row_names = ["Vote Accuracy", "Oracle Acc", "Avg Pool Size"]
    for i in range(3):
        val_base, val_final = m_base[i], m_final[i]
        if i == 2:
            print(f"{row_names[i]:<20} | {val_base:<18.1f} | {val_final:<18.1f}")
        else:
            print(f"{row_names[i]:<20} | {val_base:<17.2f}% | {val_final:<17.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluating Jitter-based voting strategy for LLM reasoning.")
    
    parser.add_argument("--input", type=str, default="gsm8k_standard_N16_scored.jsonl", help="Path to input JSONL file")
    parser.add_argument("--output", type=str, default="final_report.jsonl", help="Path to output JSONL file")
    
    parser.add_argument("--sample_s", type=int, default=0, help="Start index for candidate sampling")
    parser.add_argument("--sample_e", type=int, default=16, help="End index for candidate sampling")

    parser.add_argument("--v100_jitter", type=float, default=10, help="Top percentage threshold of jitter to filter out for high-confidence paths")
    parser.add_argument("--v0_jitter", type=float, default=90, help="Top percentage threshold of jitter to keep for low-confidence paths")
    parser.add_argument("--min_score", type=float, default=100, help="Minimum verbalized score required to be classified as high-confidence")
    
    args = parser.parse_args()
    main(args)