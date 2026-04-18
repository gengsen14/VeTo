import json
import re
import os
from tqdm import tqdm
from openai import OpenAI

# Input file
INPUT_FILE = "rbench_8B_think_N16_with_probs.jsonl" 
# Output file after fixing
OUTPUT_FILE = INPUT_FILE.replace(".jsonl", "_FIXED.jsonl")

# DeepSeek API Configuration
API_KEY = "xxxxx"
BASE_URL = "https://api.deepseek.com"

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

def extract_answer(text, task_type="GPQ"):
    """
    Extract answer for formats: \boxed{\text{D. ~ 33.4}}, \boxed{A. d}, \boxed{33.4}
    """
    if not text: return None
    text = str(text).strip()
    
    # 1. Extract Box content
    target_content = None
    idx = text.rfind("\\boxed")
    if idx != -1:
        open_idx = text.find("{", idx)
        if open_idx != -1:
            balance = 1
            for i in range(open_idx + 1, len(text)):
                char = text[i]
                if char == '{': balance += 1
                elif char == '}': balance -= 1
                if balance == 0:
                    target_content = text[open_idx + 1 : i]
                    break
    
    if not target_content:
        if "####" in text:
            target_content = text.split("####")[-1]
        else:
            target_content = text[-100:]

    if not target_content: return None

    # 2. Clean text
    clean_text = target_content
    
    # Replace special characters with spaces
    for char in ["~", "\\", "$", "%", "{", "}"]:
        clean_text = clean_text.replace(char, " ")
        
    # Remove LaTeX keywords
    for kw in ["text", "textbf", "boxed", "huge", "cal", "rm"]:
        clean_text = clean_text.replace(kw, " ")

    # 3. Extract by task type    
    if task_type == "MATH":
        # Remove commas
        math_text = clean_text.replace(",", "")
        # Extract the last number
        matches = re.findall(r"[-+]?\d*\.\d+|\d+", math_text)
        if matches: return matches[-1]
        return None

    elif task_type == "GPQ":
        upper_text = clean_text.upper().strip()

        # 1. Split tokens
        tokens = upper_text.split()
        
        # 2. Filter valid tokens
        valid_opts = []
        
        for t in tokens:
            # Remove interfering symbols
            raw_t = t.replace(".", "").replace(":", "").replace(")", "").replace("(", "")
            
            # Check if it's a single alphabetic character
            if len(raw_t) == 1 and raw_t.isalpha():
                valid_opts.append(raw_t)
        
        # 3. Decision logic
        if not valid_opts:
            return None

        return valid_opts[-1]

    return None

def normalize_answer(ans):
    """
    Normalize answer: convert to float/int string for comparison
    """
    if ans is None: return ""
    ans = str(ans).strip().replace(",", "").replace("$", "").replace(" ", "")
    try:
        # Convert to numeric type for comparison
        if "." in ans:
            return str(float(ans))
        return str(int(ans))
    except:
        return ans

def check_correctness(model_ans, gold_ans):
    return normalize_answer(model_ans) == normalize_answer(gold_ans)

def judge_with_deepseek(question, gold_answer, model_response):
    """
    Call DeepSeek to verify if the model's answer is correct and extract the answer
    """
    prompt = f"""
You are a strict Math Evaluator.
Task: Evaluate if the STUDENT'S RESPONSE matches the GOLD ANSWER.

Problem: {question}

Gold Answer: {gold_answer}

Student's Response:
{model_response}

---
Requirements:
1. Ignore the reasoning path if it contains errors, ONLY look at the FINAL ANSWER.
2. If the student eventually corrects themselves and outputs the correct final number (e.g. inside \\boxed{{}} or at the end), count it as CORRECT.
3. If the student produces multiple conflicting answers, count it as WRONG.

Output strictly in JSON format:
{{
    "extracted_answer": "the number extracted from student response",
    "is_correct": true/false
}}
"""
    try:
        response = client.chat.completions.create(
            model="deepseek-chat", # or deepseek-reasoner
            messages=[
                {"role": "system", "content": "You are a helpful assistant that outputs JSON."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        result_text = response.choices[0].message.content
        return json.loads(result_text)
    except Exception as e:
        print(f"API Error: {e}")
        return {"extracted_answer": None, "is_correct": False}


def main():
    print(f"Fixing results in: {INPUT_FILE}")
    
    if not os.path.exists(INPUT_FILE):
        print("File not found.")
        return

    fixed_count = 0
    records = []

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        # Read all lines
        lines = f.readlines()[:]

    call_count = 0

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as fout:
        for line in tqdm(lines, desc="Processing"):
            if not line.strip(): continue
            record = json.loads(line)

            # 1. Get and process Gold Answer
            question = record.get("problem") or record.get("question")
            raw_gold = record.get("gold_answer", "") or record.get("answer", "")
            gold_extracted = extract_answer(raw_gold)
            if len(raw_gold) <= 1:
                gold_extracted = raw_gold

            # 2. Re-process each candidate
            candidates = record.get("candidates", [])
            correct_count = 0
            
            for cand in candidates:
                # Compatible with different field names (text vs full_text)
                text_content = cand.get("extracted_answer", None)
                
                # Re-extract answer
                if cand.get("extracted_answer") is not None:
                    pred_ans = text_content
                else:
                    text_content = cand.get("text", cand.get("full_text", cand.get("continuation_text", "")))
                    pred_ans = extract_answer(text_content)
                
                # Re-judge correctness
                is_correct = check_correctness(pred_ans, gold_extracted) or (gold_extracted == pred_ans and pred_ans is not None)

                if pred_ans is None:
                    call_count += 1
                    print(f"currenct call {call_count}")
                    
                    judge_result = judge_with_deepseek(question, raw_gold, text_content)
                    
                    # Use LLM judgment as the standard
                    if judge_result.get("is_correct") is True:
                        pred_ans = judge_result.get("extracted_answer", pred_ans)
                        is_correct = True
                    else:
                        pred_ans = judge_result.get("extracted_answer", pred_ans)
                        is_correct = False
                
                # Update fields
                cand["extracted_answer"] = pred_ans
                cand["is_correct"] = is_correct
                
                if is_correct:
                    correct_count += 1
            
            # 3. Update statistics
            total = len(candidates)
            record["stats"] = {
                "total": total,
                "correct": correct_count,
                "accuracy": correct_count / total if total > 0 else 0.0
            }
            
            # Write to new file
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fixed_count += 1

    print(f"Fixed {fixed_count} records, call {call_count}")
    print(f"Saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
