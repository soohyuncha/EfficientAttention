"""
Fused Attention
===============

This is a Triton implementation of the Flash Attention v2 algorithm from Tri Dao (https://tridao.me/publications/flash2/flash2.pdf)

Credits: OpenAI kernel team

Extra Credits:

* Original flash attention paper (https://arxiv.org/abs/2205.14135)
* Rabe and Staats (https://arxiv.org/pdf/2112.05682v2.pdf)

"""

import torch
import flash_attn

import os
import math
import argparse
import time

import triton
import triton.language as tl

import custom_flash_attn


DTYPE = torch.float16
DEVICE = "cuda:0"


@triton.jit
def _attn_fwd_inner_v1(
        acc,                # [BLOCK_M, HEAD_DIM]
        l_i, m_i,           # [BLOCK_M]
        q,                  # [BLOCK_M, HEAD_DIM]; loaded to shared mem.
        k_blk_ptr, v_blk_ptr,   # tl.block_ptr
        mask_ptr, s_ptr,        # pointer
        q_blk_idx, qk_scale,
        offs_m, offs_n, N_CTX, TREE_SIZE,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        STAGE: tl.constexpr,
        debug: tl.constexpr):
    # Determine the range of KV blocks to iterate
    if STAGE == 1:
        lo, hi = 0, N_CTX
    else:
        lo, hi = N_CTX, N_CTX + TREE_SIZE

    k_blk_ptr = tl.advance(k_blk_ptr, (lo, 0))
    v_blk_ptr = tl.advance(v_blk_ptr, (lo, 0))

#    if debug:
#        tl.store(QQ_ptr + offs_m[:, None] * HEAD_DIM + tl.arange(0, HEAD_DIM)[None, :], q.to(tl.float32))

    # loop over k, v and update accumulator
    for start_n in tl.range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        #################################
        # 1) Compute Q*k^T
        #################################
        k = tl.load(k_blk_ptr).T            # [BLOCK_N, HEAD_DIM] -> [HEAD_DIM, BLOCK_N]
        mask = tl.load(mask_ptr + offs_m[:, None] * (N_CTX + TREE_SIZE) + (offs_n[None, :] + start_n))        # [BLOCK_M, BLOCK_N]; 1 valid, 0 invalid
        qk = tl.dot(q, k)

        # Apply mask operation only for the last few "tree" blocks
        if STAGE == 1:
            qk = qk * qk_scale
        else:
            qk = qk * qk_scale + tl.where(mask, 0, -float("inf"))

        if debug:
            s_indices = offs_m[:, None] * (N_CTX + TREE_SIZE) + (offs_n[None, :] + start_n)
            tl.store(s_ptr + s_indices, qk)

        
        #################################
        # 2) Softmax
        #################################
        m_ij = tl.maximum(m_i, tl.max(qk, 1))       # Updated "row_max"
        qk -= m_ij[:, None]
        p = tl.math.exp(qk)

        l_ij = tl.sum(p, 1)                         # "row_exp_sum" of j-th tile
        alpha = tl.math.exp(m_i - m_ij)             # correction factor

        acc = acc * alpha[:, None]

        #################################
        # 3) Compute S*V
        #################################
        v = tl.load(v_blk_ptr)
        p = p.to(tl.float16)
        acc = tl.dot(p, v, acc)

        # Update m_i and l_i
        l_i = l_i * alpha + l_ij
        m_i = m_ij

        # Move on to next KV tile
        k_blk_ptr = tl.advance(k_blk_ptr, (BLOCK_N, 0))
        v_blk_ptr = tl.advance(v_blk_ptr, (BLOCK_N, 0))

    return acc, l_i, m_i


@triton.jit
def _attn_fwd_v1(q, k, v, attn_mask,
                 o, M, S, #QQ,
                 B, H, N_CTX, TREE_SIZE, qk_scale,
                 HEAD_DIM: tl.constexpr,
                 BLOCK_M: tl.constexpr,
                 BLOCK_N: tl.constexpr,
                 debug: tl.constexpr,
                 ):
    dtype = tl.float16
    tl.static_assert(BLOCK_N <= HEAD_DIM)

    # Index of "program instance", which corresponds to "CTA" in CUDA
    q_blk_idx = tl.program_id(0)      # block id of query dim.
    bh_idx = tl.program_id(1)
    b_idx = bh_idx // H             # batch idx
    h_idx = bh_idx % H              # head idx
    
    # Make block_ptr for Q, K, V, and O
    # TODO: But why use block_ptr..?
    q_block_ptr = tl.make_block_ptr(
            base = q + (b_idx * H * TREE_SIZE * HEAD_DIM) + (h_idx * TREE_SIZE * HEAD_DIM),     # points to 1st query, 1st elem
            shape = (TREE_SIZE, HEAD_DIM),
            strides = (HEAD_DIM, 1),
            offsets = (q_blk_idx * BLOCK_M, 0),                                 # points to 1st query of the "q_blk_idx"-th block, 1st elem
            block_shape=(BLOCK_M, HEAD_DIM),
            order = (0, 1),
    )
#    mask_block_ptr = tl.make_block_ptr(
#            base = attn_mask + (b_idx * TREE_SIZE * (N_CTX + TREE_SIZE)),       # points to 1st query, 1st elem
#            shape = (TREE_SIZE, N_CTX + TREE_SIZE),
#            strides = (N_CTX + TREE_SIZE, 1),
#            offsets = (q_blk_idx * BLOCK_M, 0),                                 # points to 1st query of the "q_blk_idx"-th block, 1st elem
#            block_shape=(BLOCK_M, BLOCK_N),
#            order = (0, 1),
#    )

    k_block_ptr = tl.make_block_ptr(
            base = k + (b_idx * H * (N_CTX + TREE_SIZE) * HEAD_DIM) + (h_idx * (N_CTX + TREE_SIZE) * HEAD_DIM),      # points to 1st context token, 1st elem
            shape = (N_CTX + TREE_SIZE, HEAD_DIM),
            strides = (HEAD_DIM, 1),
            offsets = (0, 0),
            block_shape=(BLOCK_N, HEAD_DIM),
            order = (0, 1),
    )
    v_block_ptr = tl.make_block_ptr(
            base = v + (b_idx * H * (N_CTX + TREE_SIZE) * HEAD_DIM) + (h_idx * (N_CTX + TREE_SIZE) * HEAD_DIM),      # points to 1st context token, 1st elem
            shape = (N_CTX + TREE_SIZE, HEAD_DIM),
            strides = (HEAD_DIM, 1),
            offsets = (0, 0),
            block_shape=(BLOCK_N, HEAD_DIM),
            order = (0, 1),
    )
    o_block_ptr = tl.make_block_ptr(
            base = o + (b_idx * H * TREE_SIZE * HEAD_DIM) + (h_idx * TREE_SIZE * HEAD_DIM),      # points to 1st query, 1st elem
            shape = (TREE_SIZE, HEAD_DIM),
            strides = (HEAD_DIM, 1),
            offsets = (q_blk_idx * BLOCK_M, 0),                                 # points to 1st query of the "q_blk_idx"-th block, 1st elem
            block_shape=(BLOCK_M, HEAD_DIM),
            order = (0, 1)
    )
    s_ptr = S + (b_idx * H * TREE_SIZE * (N_CTX + TREE_SIZE)) + (h_idx * TREE_SIZE * (N_CTX + TREE_SIZE))      # points to 1st query, 1st elem
    mask_ptr = attn_mask + (b_idx * TREE_SIZE * (N_CTX + TREE_SIZE))             # points to 1st query, 1st elem
#    QQ_ptr = QQ + (b_idx * H * N_CTX * HEAD_DIM) + (h_idx * N_CTX * HEAD_DIM)

    # initialize offsets
    offs_m = q_blk_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")      # row max. q*k^T for each query
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)               # row sum. of e^S for each query
    o_block = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)


    # Load "q_blk_idx"-th tile into SRAM
    q_block = tl.load(q_block_ptr)
    
    o_block, l_i, m_i = _attn_fwd_inner_v1(o_block, l_i, m_i, q_block,
                                        k_block_ptr, v_block_ptr, mask_ptr, s_ptr,# QQ_ptr,
                                        q_blk_idx, qk_scale,
                                        offs_m, offs_n, N_CTX, TREE_SIZE,
                                        BLOCK_M, BLOCK_N,
                                        HEAD_DIM,
                                        1,
                                        debug)

    o_block, l_i, m_i = _attn_fwd_inner_v1(o_block, l_i, m_i, q_block,
                                        k_block_ptr, v_block_ptr, mask_ptr, s_ptr,# QQ_ptr,
                                        q_blk_idx, qk_scale,
                                        offs_m, offs_n, N_CTX, TREE_SIZE,
                                        BLOCK_M, BLOCK_N,
                                        HEAD_DIM,
                                        2,
                                        debug)

    # epilogue
    o_block = o_block / l_i[:, None]
    m_ptrs = M + bh_idx * TREE_SIZE + offs_m
    tl.store(m_ptrs, m_i)
    tl.store(o_block_ptr, o_block.to(tl.float16))



def triton_flash_attn_tree_v1(q, k, v, attn_mask, debug=False):
    ############################################
    # q: [BSZ, H, TREE_SIZE, HEAD_DIM]
    # k, v: [BSZ, H, N_CTX + TREE_SIZE, HEAD_DIM]
    # attn_mask: [bsz, 1, tree_size, N_CTX]
    ############################################

    # Shape constraints
    BSZ, H, TREE_SIZE, HEAD_DIM = q.shape
    N_CTX = k.shape[2] - TREE_SIZE

    # Output tensor
    o = torch.empty_like(q)

    M = torch.empty((BSZ, H, TREE_SIZE), device=DEVICE, dtype=torch.float32)
    S = torch.empty((BSZ, H, TREE_SIZE, N_CTX + TREE_SIZE), device=DEVICE, dtype=torch.float32)
#    QQ = torch.empty((q.shape[0], q.shape[1], q.shape[2], q.shape[3]), device=q.device, dtype=torch.float32)


    def grid(META):
        return (triton.cdiv(TREE_SIZE, META["BLOCK_M"]), BSZ * H, 1)
    
    qk_scale = 1 / math.sqrt(HEAD_DIM)
    _attn_fwd_v1[grid](
            q, k, v, attn_mask,
            o, M, S,
            BSZ, H, N_CTX, TREE_SIZE, qk_scale,
            HEAD_DIM=HEAD_DIM,
            debug=debug,
            BLOCK_M=64, BLOCK_N=64,
        )
    
    return o, S, M
