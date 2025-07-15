import torch
import flash_attn
import custom_flash_attn
from triton_src.triton_tree_attn import triton_flash_attn_tree_v1

import os
import math
import argparse
import time
import numpy as np

import triton
import triton.language as tl

DTYPE=torch.float16
DEVICE="cuda:0"

def bench_flash_attention(query, key, value, BSZ, H, N_CTX, TREE_SIZE, HEAD_DIM, tree_mask=None, NUM_TESTS=10):
    def _get_full_mask(BSZ, N_CTX, TREE_SIZE, tree_mask):
        # one_mask: [bsz, 1, tree_size, n_ctx]
        # tree_mask: [bsz, 1, tree_size, tree_size]
        one_mask = torch.ones(BSZ, 1, TREE_SIZE, N_CTX).to(tree_mask.dtype).to(tree_mask.device)
        return torch.cat((one_mask, tree_mask), dim=-1)

    attn_mask = _get_full_mask(BSZ, N_CTX, TREE_SIZE, tree_mask)        # valid: 1, invalid: 0
    attn_mask_1byte = attn_mask.to(torch.uint8)
    torch_mask = torch.where(attn_mask == 0, -float("inf"), attn_mask)
    torch_mask = torch.where(torch_mask == 1, 0, torch_mask)               # valid: 0, invalid: -inf
    
    
    # 1) Torch
    p = torch.matmul(query, key.transpose(2, 3)) * (1.0 / math.sqrt(HEAD_DIM))
    p = p + torch_mask
    p = torch.softmax(p.float(), dim = -1).to(torch.float16)
    ref_out_torch = torch.matmul(p, value)

    # 2) FA-2
    ref_out_fa2 = flash_attn.flash_attn_func(query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2), causal=True).transpose(1, 2)  

    # 4) Triton-based
    out_triton, s_triton, m_triton = triton_flash_attn_tree_v1(query, key, value, attn_mask_1byte, True)
    s_torch = torch.matmul(query, key.transpose(2, 3)) * (1.0 / math.sqrt(HEAD_DIM)) + torch_mask
    s_torch = s_torch.float()
    s_torch = torch.where(s_torch < -1e5, 0, s_torch)
    s_triton = torch.where(s_triton < -1e5, 0, s_triton)
    row_max_torch = torch.max(s_torch, dim=-1).values

    func_check = torch.allclose(s_torch, s_triton, rtol=0, atol=1e-2)
    if func_check:
        print(f'> Triton functionality for S correct!')
    else:
        print(f'> Triton functionality for S wrong...')
        print(torch.topk(abs(s_torch.reshape(-1) - s_triton.reshape(-1)), k=20, dim=0))
        exit()

    func_check = torch.allclose(row_max_torch, m_triton, rtol=0, atol=1e-2)
    if func_check:
        print(f'> Triton functionality for M correct!')
    else:
        print(f'> Triton functionality for M wrong...')
        print(torch.topk(abs(row_max_torch.reshape(-1) - m_triton.reshape(-1)), k=20, dim=0))
        exit()

    func_check = torch.allclose(ref_out_torch, out_triton, rtol=0, atol=1e-2)
    if func_check:
        print(f'> Triton functionality correct!')
    else:
        print(f'> Triton functionality wrong...')
        print(torch.topk(abs(ref_out_torch.reshape(-1) - out_triton.reshape(-1)), k=20, dim=0))
        exit()
    print()
    

    # Compute FLOPs
    GBs_per_matmul = (BSZ * H * (N_CTX + TREE_SIZE) * HEAD_DIM * 2) / (10**9)
    total_GBs = 2 * GBs_per_matmul

    # Profile starts!
    NUM_WARMUP = 3

    # 1) Torch
    torch_latency = []
    for i in range(NUM_WARMUP + NUM_TESTS):
        torch.cuda.synchronize()
        s = time.time()

        p = torch.matmul(query, key.transpose(2, 3)) * (1.0 / math.sqrt(HEAD_DIM))
        p = p + torch_mask
        p = torch.softmax(p.float(), dim = -1).to(torch.float16)
        ref_out_torch = torch.matmul(p, value)

        torch.cuda.synchronize()
        e = time.time()
        if i >= NUM_WARMUP:
            torch_latency.append((e - s) * (10**3))

    # 2) FA-2
    fa2_latency = []
    for i in range(NUM_WARMUP + NUM_TESTS):
        torch.cuda.synchronize()
        s = time.time()

        ref_out_fa2 = flash_attn.flash_attn_func(query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2), causal=True).transpose(1, 2)

        torch.cuda.synchronize()
        e = time.time()
        if i >= NUM_WARMUP:
            fa2_latency.append((e - s) * (10**3))

    # 3) Triton-based
    triton_latency = []
    for i in range(NUM_WARMUP + NUM_TESTS):
        torch.cuda.synchronize()
        s = time.time()

        out_triton, _, _ = triton_flash_attn_tree_v1(query, key, value, attn_mask_1byte, False)

        torch.cuda.synchronize()
        e = time.time()
        if i >= NUM_WARMUP:
            triton_latency.append((e - s) * (10**3))
    
    torch_latency = np.array(torch_latency)
    fa2_latency = np.array(fa2_latency)
    triton_latency = np.array(triton_latency)


    print("=========================")
    print("Profile result")
    print("=========================")
    print(f"> Ideal: {(total_GBs / 1008) * 1000: .2f} ms \t 1,008 GB/s")
    print(f"> Mean")
    print(f"\t > Torch: {np.mean(torch_latency): .3f} ms \t {total_GBs / (np.mean(torch_latency) / (10**3)): .2f} GB/s")
    print(f"\t > FA-2 (CUDA): {np.mean(fa2_latency): .3f} ms \t {total_GBs / (np.mean(fa2_latency) / (10**3)): .2f} GB/s")
    print(f"\t > Triton: {np.mean(triton_latency): .3f} ms \t {total_GBs / (np.mean(triton_latency) / (10**3)): .2f} GB/s")

    print(f"> Median")
    print(f"\t > Torch: {np.median(torch_latency): .3f} ms \t {total_GBs / (np.median(torch_latency) / (10**3)): .2f} GB/s")
    print(f"\t > FA-2 (CUDA): {np.median(fa2_latency): .3f} ms \t {total_GBs / (np.median(fa2_latency) / (10**3)): .2f} GB/s")
    print(f"\t > Triton: {np.median(triton_latency): .3f} ms \t {total_GBs / (np.median(triton_latency) / (10**3)): .2f} GB/s")


def generate_random_tree(bsz, tree_size):
    np.random.seed(1234)

    mask = np.zeros((bsz, 1, tree_size, tree_size), dtype=int)

    mask[:, :, 0, 0] = 1

    for b in range(bsz):
        parent = [-1] * tree_size
        
        for i in range(1, tree_size):
            p = np.random.randint(0, i)
            parent[i] = p
            mask[b, :, i] = mask[b, :, p].copy()
            mask[b, :, i, i] = 1

    mask = torch.tensor(mask).to(DEVICE)
    return mask

def print_tree_mask(mask):
    for b in range(mask.shape[0]):
        for i in range(mask.shape[2]):
            for j in range(mask.shape[3]):
                if mask[b, 0, i, j] == -float("inf"):
                    print("x", end=' ')
                else:
                    print(mask[b, 0, i, j].to(torch.int).item(), end=' ')
            print()
        print()
    print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--b", type=int, default=8, help="Batch size",)
    parser.add_argument("--h", type=int, default=32, help="#. of heads",)
    parser.add_argument("--n", type=int, default=1024, help="Query length",)
    parser.add_argument("--dim", type=int, default=128, help="Head dim",)
    parser.add_argument("--tree_size", type=int, default=64, help="Tree size",)
    parser.add_argument("--num_tests", type=int, default=10, help="Iteration number",)

    args = parser.parse_args()

    b, h, n, dim, tree_size = args.b, args.h, args.n, args.dim, args.tree_size

    torch.manual_seed(1234)

    query = torch.rand((b, h, tree_size, dim), dtype=DTYPE, device=DEVICE)
    key = torch.rand((b, h, n + tree_size, dim), dtype=DTYPE, device=DEVICE)
    value = torch.rand((b, h, n + tree_size, dim), dtype=DTYPE, device=DEVICE)

    tree_mask = generate_random_tree(b, tree_size)
#    print_tree_mask(tree_mask)
#    exit()

    bench_flash_attention(query, key, value, b, h, n, tree_size, dim, tree_mask=tree_mask, NUM_TESTS=args.num_tests)
