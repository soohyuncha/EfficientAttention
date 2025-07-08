# EfficientAttention

```
pip install -e .
# v1: online softmax, thread block assignment
python benchmark.py --b 8 --h 32 --n 1024 --dim 128 --version 1
# v2: WMMA instruction instead of FMA
python benchmark.py --b 8 --h 32 --n 1024 --dim 128 --version 2
# v3: Optimization to reduce unnecessary compute & global mem. access proposed in FlashAttn-2
python benchmark.py --b 8 --h 32 --n 1024 --dim 128 --version 3
```

## Categorize optimization techniques
| Technique                  | V1 | V2 | V3 | V4 |
|:--------------------------:|:--:|:--:|:--:|:--:|
|Online softmax (FlashAttn-1)| O | O | O | O |
|Thread-block (FlashAttn-2)  | O | O | O | O |
|WMMA instruction            | | O | O | O |
|Maximize shared mem. for output(FlashAttn-2)| | | O | O |
|Vectorized load/store| | | | O |
