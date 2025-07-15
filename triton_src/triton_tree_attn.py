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
def _attn_fwd_inner(
        acc, l_i, m_i, q,
        k_blk_ptr, v_blk_ptr, s_ptr, #QQ_ptr,
        q_blk_idx, qk_scale,
        offs_m, offs_n,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
        N_CTX: tl.constexpr, HEAD_DIM: tl.constexpr,
        STAGE: tl.constexpr,
        warp_specialize: tl.constexpr, debug: tl.constexpr):
    # Determine the range of KV blocks to iterate
    # 1) Dense part
    if STAGE == 1:
        lo, hi = 0, q_blk_idx * BLOCK_M
    # 2) Causal part (The last block)
    elif STAGE == 2:
        lo, hi = q_blk_idx * BLOCK_M, (q_blk_idx + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
    # causal = False
    else:
        lo, hi = 0, N_CTX

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
        k = tl.load(k_blk_ptr).T
        
        qk = tl.dot(q, k)
        if STAGE == 2:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = qk * qk_scale + tl.where(mask, 0, -float("inf"))
        else:
            qk = qk * qk_scale

        if debug:
            s_indices = offs_m[:, None] * N_CTX + (offs_n[None, :] + start_n)
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
def _attn_fwd(q, k, v, o,       # [B, H, N_CTX, HEAD_DIM]
              M, S, #QQ,
              B, H, qk_scale,
              N_CTX: tl.constexpr,
              HEAD_DIM: tl.constexpr,
              BLOCK_M: tl.constexpr,
              BLOCK_N: tl.constexpr,
              STAGE: tl.constexpr,
              warp_specialize: tl.constexpr,
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
            base = q + (b_idx * H * N_CTX * HEAD_DIM) + (h_idx * N_CTX * HEAD_DIM),      # points to 1st query of the "q_blk_idx"th block, 1st elem
            shape = (N_CTX, HEAD_DIM),
            strides = (HEAD_DIM, 1),
            offsets = (q_blk_idx * BLOCK_M, 0),
            block_shape=(BLOCK_M, HEAD_DIM),
            order = (0, 1),
    )
    k_block_ptr = tl.make_block_ptr(
            base = k + (b_idx * H * N_CTX * HEAD_DIM) + (h_idx * N_CTX * HEAD_DIM),      # points to 1st key, 1st elem
            shape = (N_CTX, HEAD_DIM),
            strides = (HEAD_DIM, 1),
            offsets = (0, 0),
            block_shape=(BLOCK_N, HEAD_DIM),
            order = (0, 1),
    )
    v_block_ptr = tl.make_block_ptr(
            base = v + (b_idx * H * N_CTX * HEAD_DIM) + (h_idx * N_CTX * HEAD_DIM),      # points to 1st value, 1st elem
            shape = (N_CTX, HEAD_DIM),
            strides = (HEAD_DIM, 1),
            offsets = (0, 0),
            block_shape=(BLOCK_N, HEAD_DIM),
            order = (0, 1),
    )
    o_block_ptr = tl.make_block_ptr(
            base = o + (b_idx * H * N_CTX * HEAD_DIM) + (h_idx * N_CTX * HEAD_DIM),      # points to 1st query of the "q_blk_idx"th block, 1st elem
            shape = (N_CTX, HEAD_DIM),
            strides = (HEAD_DIM, 1),
            offsets = (q_blk_idx * BLOCK_M, 0),
            block_shape=(BLOCK_M, HEAD_DIM),
            order = (0, 1)
    )
    s_ptr = S + (b_idx * H * N_CTX * N_CTX) + (h_idx * N_CTX * N_CTX)      # points to 1st query of the "q_blk_idx"th block, 1st elem
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
    
    # Dense part
    if STAGE & 1:
        o_block, l_i, m_i = _attn_fwd_inner(o_block, l_i, m_i, q_block,
                                            k_block_ptr, v_block_ptr, s_ptr,# QQ_ptr,
                                            q_blk_idx, qk_scale,
                                            offs_m, offs_n,
                                            BLOCK_M, BLOCK_N,
                                            N_CTX, HEAD_DIM,
                                            4 - STAGE,
                                            warp_specialize, debug)

    # Causal part
    if STAGE & 2:
        o_block, l_i, m_i = _attn_fwd_inner(o_block, l_i, m_i, q_block,
                                            k_block_ptr, v_block_ptr, s_ptr,# QQ_ptr,
                                            q_blk_idx, qk_scale,
                                            offs_m, offs_n,
                                            BLOCK_M, BLOCK_N,
                                            N_CTX, HEAD_DIM,
                                            2,
                                            warp_specialize, debug)

    # epilogue
    o_block = o_block / l_i[:, None]
    m_ptrs = M + bh_idx * N_CTX + offs_m
    tl.store(m_ptrs, m_i)
    tl.store(o_block_ptr, o_block.to(tl.float16))



def triton_flash_attn_prefill(q, k, v, causal, warp_specialize=True, debug=False):
    # shape constraints
    HEAD_DIM = q.shape[-1]

    o = torch.empty_like(q)
    stage = 3 if causal else 1

    M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
    S = torch.empty((q.shape[0], q.shape[1], q.shape[2], q.shape[2]), device=q.device, dtype=torch.float32)
#    QQ = torch.empty((q.shape[0], q.shape[1], q.shape[2], q.shape[3]), device=q.device, dtype=torch.float32)

    B, H = q.shape[0], q.shape[1]

    def grid(META):
        return (triton.cdiv(q.shape[2], META["BLOCK_M"]), B * H, 1)
    
    qk_scale = 1 / math.sqrt(HEAD_DIM)
    _attn_fwd[grid](
            q, k, v, o,
            M, S,
            B, H, qk_scale,
            N_CTX=q.shape[2],
            HEAD_DIM=HEAD_DIM,
            STAGE=stage,
            warp_specialize=warp_specialize,
            debug=debug,
            BLOCK_M=64,
            BLOCK_N=64,
        )
    
    return o, S, M
