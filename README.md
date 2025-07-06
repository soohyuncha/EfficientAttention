# EfficientAttention

```
pip install -e .
# v1: online softmax, thread block assignment
python benchmark.py --b 8 --h 32 --n 1024 --dim 128 --version 1
# v2: WMMA instruction instead of FMA
python benchmark.py --b 8 --h 32 --n 1024 --dim 128 --version 2
```

## Categorize optimization techniques
| Technique                  | V1 | V2 | V3 |
|:--------------------------:|:--:|:--:|:--:|
|Online softmax (FlashAttn-1)| O | O | O |
|Thread-block (FlashAttn-2)  | O | O | O |
|WMMA instruction            | | O | O |
|Maximize shared mem. (FlashAttn-2)| |  | O |
