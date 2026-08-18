"""
NoVictim: Hardware-Aligned Block Coding for Outlier-Aware W2A4
================================================================

核心想法：
  - 一個 block = 16 weights = 32 bits（硬約束，平均 2 bit/weight）
  - 普通 weight 用 ternary {-1, 0, +1} 表示
  - 每個 block 最多一個 outlier，給更多 reconstruction levels
  - 沒有 victim：Case B 的 15 個 ternary 全部保留
  - flag 融進「17 態位置編碼」：position=16 表示無 outlier
"""

import numpy as np

BLOCK = 16
POS_STATES = 17
NO_OUTLIER = 16

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
    # 只用 15 個非零值 + 0，共 15 個，index 0-14
    return np.array(values[:15], dtype=np.float64)

OUTLIER_CODEBOOK = build_outlier_codebook()

def pack_ternary(tern):
    packed = 0
    for i, t in enumerate(tern):
        packed += int(t) * (3 ** i)
    return packed

def unpack_ternary(packed, n):
    tern = []
    for _ in range(n):
        tern.append(packed % 3)
        packed //= 3
    return tern

def to_internal(t):
    return int(t) + 1

def from_internal(t):
    return int(t) - 1

def encode_block(weights, outlier_threshold=32.0):
    weights = np.asarray(weights, dtype=np.float64)
    assert len(weights) == BLOCK
    abs_w = np.abs(weights)
    outlier_mask = abs_w > outlier_threshold

    if not outlier_mask.any():
        position = NO_OUTLIER
        tern = [to_internal(int(np.sign(w))) for w in weights]
        ternary_packed = pack_ternary(tern)
        code = position + POS_STATES * ternary_packed
        meta = {"case": "A", "position": position}
    else:
        position = int(abs_w.argmax())
        outlier_val = weights[position]
        outlier_code = int(np.argmin(np.abs(OUTLIER_CODEBOOK - outlier_val)))
        tern = []
        for i, w in enumerate(weights):
            if i == position:
                continue
            tern.append(to_internal(int(np.sign(w))))
        ternary_packed = pack_ternary(tern)
        code = position + POS_STATES * (outlier_code + 15 * ternary_packed)
        meta = {"case": "B", "position": position,
                "outlier_code": outlier_code,
                "outlier_recon": float(OUTLIER_CODEBOOK[outlier_code])}

    assert code < 2 ** 32, f"code {code} exceeds 32-bit!"
    return code, meta

def decode_block(code):
    position = code % POS_STATES
    rest = code // POS_STATES
    weights = np.zeros(BLOCK, dtype=np.float64)

    if position == NO_OUTLIER:
        tern = unpack_ternary(rest, BLOCK)
        for i in range(BLOCK):
            weights[i] = from_internal(tern[i])
    else:
        outlier_code = rest % 15
        ternary_packed = rest // 15
        tern = unpack_ternary(ternary_packed, BLOCK - 1)
        ti = 0
        for i in range(BLOCK):
            if i == position:
                weights[i] = OUTLIER_CODEBOOK[outlier_code]
            else:
                weights[i] = from_internal(tern[ti])
                ti += 1
    return weights

if __name__ == "__main__":
    print("NoVictim Codec — Round-trip Test")
    rng = np.random.default_rng(0)
    n_tests = 100000
    fails = 0
    for _ in range(n_tests):
        w = rng.uniform(-1, 1, BLOCK)
        if rng.random() < 0.5:
            pos = rng.integers(0, BLOCK)
            w[pos] = rng.choice([-1, 1]) * rng.uniform(40, 380)
        code, meta = encode_block(w)
        recon = decode_block(code)
        ok = True
        for i in range(BLOCK):
            if meta["case"] == "B" and i == meta["position"]:
                if abs(recon[i] - meta["outlier_recon"]) > 1e-9:
                    ok = False
            else:
                expected = int(np.sign(w[i])) if not (meta["case"] == "B" and i == meta["position"]) else None
                if expected is not None and abs(recon[i] - expected) > 1e-9:
                    ok = False
        if not ok:
            fails += 1
    print(f"測試 {n_tests} 次，失敗 {fails} 次")
    print("✅ PASS" if fails == 0 else "❌ FAIL")
    max_A = POS_STATES * (3 ** BLOCK) - 1
    max_B = 14 + POS_STATES * (14 + 15 * (3 ** (BLOCK-1) - 1))
    print(f"Case A 最大 code: {max_A:,} ({'< 2^32 ✓' if max_A < 2**32 else '❌'})")
    print(f"Case B 最大 code: {max_B:,} ({'< 2^32 ✓' if max_B < 2**32 else '❌'})")
