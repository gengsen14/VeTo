# VeTo

## Dataset
Our experiments evaluate models across both text-based and multimodal reasoning benchmarks. 

* The text-based reasoning datasets used in our experiments are included directly in this repository.
* To evaluate multimodal reasoning capabilities, we utilize the following widely recognized public benchmarks: [MathVista](https://huggingface.co/datasets/AI4Math/MathVista) and [MMMU](https://huggingface.co/datasets/MMMU/MMMU).


## Requirements
- `python3`
- `conda create --name env`
- `pip3 install -r requirements.txt`

## File Structure

## Workflow

1. Reasoning Path Sampling: `build_data.py` — Samples multiple reasoning trajectories from the models.
2. Answer Extraction & Refinement: `fix_judge.py` — Standardizes and extracts final answers using LLM post-processing.
3. Verbalized Confidence Generation: `get_verbalized_confidence`.py — Obtains verbalized confidence scores for each reasoning path.
4. VeTo Analysis: `veto.py` — Performs the core internal dynamics analysis and final trajectory selection.
