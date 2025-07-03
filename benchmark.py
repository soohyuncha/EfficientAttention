import torch
import torch.nn.functional as F
import math
import custom_flash_attn
import flash_attn

import time
import numpy as np
import argparse

def get_causal_mask(n):
    causal_mask = torch.tril(torch.ones((n, n), dtype=query.dtype, device=query.device))
    causal_mask = torch.where(causal_mask == 0, -torch.inf, causal_mask)
    causal_mask = torch.where(causal_mask == 1, 0, causal_mask)
    return causal_mask.unsqueeze(0).unsqueeze(0)

def torch_attn(query, key, value, mask, dim):
#    query, key, value = query.to(torch.float32), key.to(torch.float32), value.to(torch.float32)
    S = torch.matmul(query, key.transpose(2, 3)) * (1.0 / math.sqrt(dim))
    S = S + mask
    row_max = torch.max(S, dim=-1).values
    logits = F.softmax(S, dim=-1)
    out = torch.matmul(logits, value)
    return S.to(torch.float32), row_max.to(torch.float32), out

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--b", type=int, default=2, help="Batch size",)
    parser.add_argument("--h", type=int, default=32, help="#. of heads",)
    parser.add_argument("--n", type=int, default=1024, help="Query length",)
    parser.add_argument("--dim", type=int, default=128, help="Head dim",)
    parser.add_argument("--kv_len", type=int, default=0, help="KV len",)
    parser.add_argument("--version", type=int, default=1, help="Custom attn version",)


    args = parser.parse_args()
    
    b, h, n, dim = args.b, args.h, args.n, args.dim

    torch.manual_seed(1234)
    query = torch.rand((b, h, n, dim), dtype=torch.float16, device='cuda:0')
    key = torch.rand((b, h, n, dim), dtype=torch.float16, device='cuda:0')
    value = torch.rand((b, h, n, dim), dtype=torch.float16, device='cuda:0')
    causal_mask = get_causal_mask(n)

    S_torch, row_max_torch, out_torch = torch_attn(query, key, value, causal_mask, dim)

    out, row_max, _, S = custom_flash_attn.flash_attn_fp16(query, key, value, True, True, args.version)
#    print(_[0, 0, :10])
#    exit()
    out_flash = flash_attn.flash_attn_func(query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2), causal=True).transpose(1, 2)
    
    row_max = row_max.reshape(row_max.shape[0], row_max.shape[1], -1)[:, :, :n]
    
    # Logits check
    S_torch = torch.where(S_torch < -1e6, 0, S_torch)
    S = torch.where(S < -1e6, 0, S)

    print(f'=== Checking functionality of logits ===')
    print(f'\t > Result: {torch.allclose(S_torch, S, rtol=0, atol=1e-2)} at error < {1e-2}')
    if not torch.allclose(S_torch, S, rtol=0, atol=1e-2):
        print(torch.topk(abs(S_torch.reshape(-1) - S.reshape(-1)), k=20, dim=0).values)
     
    # Row-max check
    print(f'=== Checking functionality of row max ===')
    print(f'\t > Result: {torch.allclose(row_max_torch, row_max, rtol=0, atol=1e-2)} at error < {1e-2}')
    if not torch.allclose(row_max_torch, row_max, rtol=0, atol=1e-2):
        print(torch.topk(abs(row_max_torch.reshape(-1) - row_max.reshape(-1)), k=20, dim=0))

    # Out check
    print(f'=== Checking functionality of out ===')
    print(f'\t > Result: {torch.allclose(out_torch, out, rtol=0, atol=1e-2)} at error < {1e-2}')
    if not torch.allclose(out_torch, out, rtol=0, atol=1e-2):
        print(torch.topk(abs(out_torch.reshape(-1) - out.reshape(-1)), k=20, dim=0))
    
    # Out check
    print(f'=== Checking functionality of out for flash-attn ===')
    print(f'\t > Result: {torch.allclose(out_torch, out_flash, rtol=0, atol=1e-2)} at error < {1e-2}')
    if not torch.allclose(out_torch, out_flash, rtol=0, atol=1e-2):
        print(torch.topk(abs(out_torch.reshape(-1) - out_flash.reshape(-1)), k=20, dim=0))



    torch.cuda.synchronize()
    l_torch = []
    for i in range(10):
        s = time.time()
        
        torch.cuda.nvtx.range_push("Torch")
        S_torch, row_max_torch, out_torch = torch_attn(query, key, value, causal_mask, dim)
        torch.cuda.nvtx.range_pop()

        torch.cuda.synchronize()
        e = time.time()
        
        l_torch.append((e-s) * 1000)

    torch.cuda.synchronize()
    l_flash = []
    for i in range(10):
        s = time.time()
        
        torch.cuda.nvtx.range_push("Flash")
        out_flash = flash_attn.flash_attn_func(query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2), causal=True).transpose(1, 2)
        torch.cuda.nvtx.range_pop()

        torch.cuda.synchronize()
        e = time.time()
        
        l_flash.append((e-s) * 1000)

    torch.cuda.synchronize()
    l_custom = []
    for i in range(10):
        s = time.time()

        torch.cuda.nvtx.range_push("Custom")
        out, row_max, _, S = custom_flash_attn.flash_attn_fp16(query, key, value, False, False, args.version)
        torch.cuda.nvtx.range_pop()

        torch.cuda.synchronize()
        e = time.time()
        
        l_custom.append((e-s) * 1000)
    
    print(f'=== Execution time ===')
    print(f'\t Torch: {np.mean(np.array(l_torch)): .3f} ms')
    print(f'\t Flash-Attn library: {np.mean(np.array(l_flash)): .3f} ms')
    print(f'\t Custom Flash-Attn: {np.mean(np.array(l_custom)): .3f} ms')

    total_flops = (2 * b * h * n * n * dim) * 2
    print(f'\t Total required GFLOPs: {total_flops / (2**30): .2f} ')
