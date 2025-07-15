# EfficientAttention

```
pip install -e csrc
python benchmark_prefill.py --b 8 --h 32 --n 1024 --dim 128 --num_tests 10
python benchmark_tree.py --b 8 --h 32 --n 1024 --dim 128 --tree_size 64 --num_tests 10
```
## Achieved TFLOPS for prefill attention
| PyTorch | FA2 <br>(CUDA, Library) | FA2 <br>(CUDA, w/o CuTe) | FA2 <br>(Triton) | Ideal |
|:-------:|:-----------------------:|:------------------------:|:----------------:|:-----:|
| 6.9 | 154.1 | 27.0 | 110.1 | 165.2 |

## Achieved GB/s for tree attention
| PyTorch | FA2 <br>(CUDA, Library) | FA2 <br>(Triton) | Ideal |
|:-------:|:-----------------------:|:----------------:|:-----:|
| 194| 806 | 782 | 1008 |

## Categorize optimization techniques (CUDA)
|Version| Technique                  | vs Torch | vs FA-2 |
|:-----:|:--------------------------:|:--------:|:-------:|
|  v1  | Online softmax, thread-block assignment| 0.06x | 0.00x |
|  v2  | WMMA instead of FMA | 0.23x | 0.02x |
|  v3  | Shared mem. usage for intermediate results | 0.95x | 0.08x |
|  v4  | Vectorized load/store | 1.04x | 0.08x |
|  v5  | Warp primitive for max/reduction | 1.22x | 0.10x |
|  v6  | Skip non-causal attention | 2.26x | 0.18x |
|  v7  | Remove unnecessary thread sync. | 2.38x | 0.19x |
