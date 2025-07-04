# EfficientAttention

```
pip install -e .
# v1: online softmax, thread block assignment
python benchmark.py --b 8 --h 32 --n 1024 --dim 128 --version 1
# v2: WMMA instruction instead of FMA
python benchmark.py --b 8 --h 32 --n 1024 --dim 128 --version 2
```
