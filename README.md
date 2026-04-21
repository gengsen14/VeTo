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
```text
VeTo/
├── Code/
│   ├── build_data.py                      # Reasoning Path Sampling
│   ├── build_data_vl.py                   # Reasoning Path Sampling
│   ├── fix_judge.py                       # Answer Extraction & Refinement
│   ├── get_verbalized_confidence.py       # Verbalized Confidence Generation
│   ├── get_verbalized_confidence_vl.py    # Verbalized Confidence Generation
│   └── veto.py                            # VeTo Analysis
├── Configs/
│   └── requirements.txt
├── Controlled datasets/                   # Benchmark data with misleading prefixes
└── Datasets/                              # Benchmark data
```

## Workflow

1. **`build_data[_vl].py`**: Generates multiple reasoning trajectories for text and vision-language models.
2. **`fix_judge.py`**: Standardizes raw outputs and extracts final answers via LLM-based post-processing.
3. **`get_verbalized_confidence[_vl].py`**: Obtains verbalized confidence for each reasoning trajectory.
4. **`veto.py`**: Performs the core internal jitter analysis and final trajectory selection.
