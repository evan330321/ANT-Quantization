"""
NoVictim: Hardware-Aligned Block Coding for Outlier-Aware W2A4
向量化版本 v2 — 用 numpy 矩陣運算取代 Python loop，速度提升 ~1000x

介面：
  encode_block(weights)         → 單一 block（向後相容）
  decode_block(code)            → 單一 block（向後相容）
  encode_blocks_batch(flat)     → 向量化，處理所有 block
  decode_blocks_batch(codes)    → 向量化，還原所有 block
"""

import numpy as np

BLOCK = 16
POS_STATES = 17
NO_OUTLIER = 16
OUTLIER_THRESHOLD = 32.0

# ---------------------------------------------------------------------------
# Outlier codebook (E2M1 + sign, 15 個值)
# ---------------------------------------------------------------------------
def build_outlier_codebook():
    exp_bit = 2
    exp_base = 5
    mant_bit = 1
    values = [0.0]
    for i in range(exp_base, exp_base + 2 ** exp_bit):
        for j in range(int(2 ** mant_bit)):
            if i == exp_base and j == 0:
                continue
            v = 2 ** i * (1 + 2 ** (-mant_bit) * j)
            values.append(v)
            values.append(-v)
    values = sorted(set(values))
    return np.array(values[:15], dtype=np.float64)  # 15 個值，index 0-14

OUTLIER_CODEBOOK = build_outlier_codebook()
N_OUTLIER_CODES = len(OUTLIER_CODEBOOK)  # 15

# mixed-radix 用到的 power 陣列
POWERS_16 = (3 ** np.arange(BLOCK)).astype(np.int64)        # [1,3,9,...,3^15]
POWERS_15 = (3 ** np.arange(BLOCK - 1)).astype(np.int64)    # [1,3,9,...,3^14]


# ---------------------------------------------------------------------------
# 向量化 encode（核心）
# ---------------------------------------------------------------------------
def encode_blocks_batch(flat: np.ndarray, outlier_threshold: float = OUTLIER_THRESHOLD) -> np.ndarray:
    """
    flat: 1D array，長度必須是 BLOCK 的倍數（外部 padding 好）
    回傳: codes，shape (num_blocks,)，dtype int64
    """
    flat = flat.astype(np.float64)
    num_blocks = len(flat) // BLOCK
    blocks = flat.reshape(num_blocks, BLOCK)          # (num_blocks, 16)

    # ---- sign → ternary internal (0/1/2) ----
    signs = (np.sign(blocks).astype(np.int64) + 1)   # (num_blocks, 16), 值 0/1/2

    # ---- 找 outlier ----
    abs_blocks = np.abs(blocks)
    has_outlier = (abs_blocks > outlier_threshold).any(axis=1)  # (num_blocks,)
    outlier_pos = abs_blocks.argmax(axis=1)                      # (num_blocks,)

    # ---- Case A: 無 outlier ----
    # 16 個 ternary packed
    ternary_packed_16 = signs @ POWERS_16              # (num_blocks,)
    code_A = NO_OUTLIER + POS_STATES * ternary_packed_16  # (num_blocks,)

    # ---- Case B: 有 outlier ----
    # outlier value → 找最近的 codebook index
    outlier_vals = blocks[np.arange(num_blocks), outlier_pos]   # (num_blocks,)
    # distance matrix: (num_blocks, 15)
    dist = np.abs(outlier_vals[:, None] - OUTLIER_CODEBOOK[None, :])
    outlier_code = dist.argmin(axis=1)                           # (num_blocks,)

    # 15 個 ternary（去掉 outlier 位置）
    # 用 mask 取出每個 block 的 15 個值
    row_idx = np.arange(num_blocks)
    # 建立 (num_blocks, 16) bool mask，outlier 位置 = False
    keep_mask = np.ones((num_blocks, BLOCK), dtype=bool)
    keep_mask[row_idx, outlier_pos] = False
    # masked 後 reshape 成 (num_blocks, 15)
    tern_15 = signs[keep_mask].reshape(num_blocks, BLOCK - 1)   # (num_blocks, 15)
    ternary_packed_15 = tern_15 @ POWERS_15                      # (num_blocks,)

    code_B = (outlier_pos +
              POS_STATES * (outlier_code +
                            N_OUTLIER_CODES * ternary_packed_15))

    # ---- 合併 Case A / B ----
    codes = np.where(has_outlier, code_B, code_A)

    assert (codes < 2**32).all(), "有 code 超過 32-bit！"
    return codes.astype(np.int64)


# ---------------------------------------------------------------------------
# 向量化 decode（核心）
# ---------------------------------------------------------------------------
def decode_blocks_batch(codes: np.ndarray) -> np.ndarray:
    """
    codes: shape (num_blocks,)
    回傳: flat array，shape (num_blocks * BLOCK,)
    """
    codes = codes.astype(np.int64)
    num_blocks = len(codes)
    out = np.zeros((num_blocks, BLOCK), dtype=np.float64)

    position = codes % POS_STATES      # (num_blocks,)
    rest = codes // POS_STATES         # (num_blocks,)

    is_A = (position == NO_OUTLIER)    # (num_blocks,) bool
    is_B = ~is_A

    # ---- Case A decode ----
    if is_A.any():
        packed_A = rest[is_A]           # (n_A,)
        n_A = packed_A.shape[0]
        tern_A = np.zeros((n_A, BLOCK), dtype=np.int64)
        p = packed_A.copy()
        for i in range(BLOCK):
            tern_A[:, i] = p % 3
            p //= 3
        out[is_A] = (tern_A - 1).astype(np.float64)   # 0/1/2 → -1/0/+1

    # ---- Case B decode ----
    if is_B.any():
        pos_B = position[is_B]          # (n_B,)
        rest_B = rest[is_B]             # (n_B,)
        n_B = rest_B.shape[0]

        outlier_code_B = rest_B % N_OUTLIER_CODES       # (n_B,)
        packed_B = rest_B // N_OUTLIER_CODES             # (n_B,)

        # unpack 15 個 ternary
        tern_15 = np.zeros((n_B, BLOCK - 1), dtype=np.int64)
        p = packed_B.copy()
        for i in range(BLOCK - 1):
            tern_15[:, i] = p % 3
            p //= 3

        # 插回 16 個位置
        blocks_B = np.zeros((n_B, BLOCK), dtype=np.float64)
        row_idx = np.arange(n_B)

        # 建立 keep_mask: outlier 位置 = False
        keep_mask = np.ones((n_B, BLOCK), dtype=bool)
        keep_mask[row_idx, pos_B] = False

        # 填入 ternary（非 outlier 位置）
        blocks_B[keep_mask] = (tern_15 - 1).astype(np.float64).reshape(-1)

        # 填入 outlier
        blocks_B[row_idx, pos_B] = OUTLIER_CODEBOOK[outlier_code_B]

        out[is_B] = blocks_B

    return out.reshape(-1)


# ---------------------------------------------------------------------------
# 單一 block 介面（向後相容，供測試用）
# ---------------------------------------------------------------------------
def encode_block(weights, outlier_threshold=OUTLIER_THRESHOLD):
    weights = np.asarray(weights, dtype=np.float64)
    assert len(weights) == BLOCK
    codes = encode_blocks_batch(weights)
    code = int(codes[0])

    # 重建 meta
    position = code % POS_STATES
    if position == NO_OUTLIER:
        meta = {"case": "A", "position": position}
    else:
        rest = code // POS_STATES
        outlier_code = rest % N_OUTLIER_CODES
        meta = {"case": "B", "position": position,
                "outlier_code": outlier_code,
                "outlier_recon": float(OUTLIER_CODEBOOK[outlier_code])}
    return code, meta


def decode_block(code):
    codes = np.array([code], dtype=np.int64)
    return decode_blocks_batch(codes)


# ---------------------------------------------------------------------------
# Round-trip 測試
# ---------------------------------------------------------------------------
def _test_roundtrip():
    print("=" * 60)
    print("NoVictim Codec v2 (向量化) — Round-trip Test")
    print("=" * 60)

    rng = np.random.default_rng(0)
    n_tests = 100000
    BSIZE = BLOCK

    # 一次造所有 block
    w = rng.uniform(-1, 1, (n_tests, BSIZE))
    # 50% 機率有 outlier
    has_out = rng.random(n_tests) < 0.5
    pos_out = rng.integers(0, BSIZE, n_tests)
    sign_out = rng.choice([-1, 1], n_tests)
    mag_out = rng.uniform(40, 380, n_tests)
    w[has_out, pos_out[has_out]] = sign_out[has_out] * mag_out[has_out]

    flat = w.reshape(-1)

    import time
    t0 = time.time()
    codes = encode_blocks_batch(flat)
    recon = decode_blocks_batch(codes)
    t1 = time.time()

    print(f"\n向量化處理 {n_tests} 個 block 耗時: {t1-t0:.3f} 秒")

    # 驗證
    recon_blocks = recon.reshape(n_tests, BSIZE)
    position = codes % POS_STATES

    fails = 0
    for b in range(n_tests):
        pos = position[b]
        if pos == NO_OUTLIER:
            # Case A: sign 一致
            for i in range(BSIZE):
                expected = int(np.sign(w[b, i]))
                if abs(recon_blocks[b, i] - expected) > 1e-9:
                    fails += 1
                    break
        else:
            # Case B: outlier 位置值正確，其他 sign 一致
            for i in range(BSIZE):
                if i == pos:
                    # outlier: 找最近 codebook
                    expected_code = int(np.argmin(np.abs(OUTLIER_CODEBOOK - w[b, i])))
                    expected_val = OUTLIER_CODEBOOK[expected_code]
                    if abs(recon_blocks[b, i] - expected_val) > 1e-9:
                        fails += 1
                        break
                else:
                    expected = int(np.sign(w[b, i]))
                    if abs(recon_blocks[b, i] - expected) > 1e-9:
                        fails += 1
                        break

    print(f"round-trip 失敗: {fails}/{n_tests}")
    print(f"結果: {'✅ PASS' if fails == 0 else '❌ FAIL'}")

    # bit budget 驗證
    max_A = POS_STATES * (3 ** BLOCK) - 1
    max_B = (POS_STATES - 1) + POS_STATES * ((N_OUTLIER_CODES - 1) + N_OUTLIER_CODES * (3 ** (BLOCK-1) - 1))
    print(f"\nCase A 最大 code: {max_A:,} ({'< 2^32 ✓' if max_A < 2**32 else '❌'})")
    print(f"Case B 最大 code: {max_B:,} ({'< 2^32 ✓' if max_B < 2**32 else '❌'})")

    # 速度對比
    print(f"\n速度估算:")
    per_block_us = (t1-t0) / n_tests * 1e6
    print(f"  向量化: {per_block_us:.4f} μs/block")
    print(f"  Python loop 估算: ~10-100 μs/block")
    print(f"  加速比: ~{int(50/per_block_us)}x")


if __name__ == "__main__":
    _test_roundtrip()