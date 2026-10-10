#!/usr/bin/env python3
"""Check for td2_8 §5: the numpy bfloat16 rounding used in the notebook matches torch,
and the spacing arithmetic of Exercise 5.1(1)."""
import numpy as np
import torch


def to_bf16(a):
    b = np.asarray(a, dtype=np.float32).view(np.uint32).astype(np.uint64)
    b = (b + 0x7FFF + ((b >> 16) & 1)) >> 16
    return (b.astype(np.uint32) << 16).view(np.float32)


rng = np.random.default_rng(0)
z = np.concatenate([rng.normal(size=200_000) * 10.0 ** rng.integers(-8, 4, 200_000),
                    rng.normal(0, 0.02, 200_000)]).astype(np.float32)
ref = torch.from_numpy(z).to(torch.bfloat16).float().numpy()
assert np.array_equal(to_bf16(z), ref)
print("numpy bf16 rounding == torch bf16 on 400k values  OK")

# spacing for 2^-6 <= |w| < 2^-5 is 2^-13
w = np.float32(0.02)
nxt = torch.nextafter(torch.tensor(0.02, dtype=torch.bfloat16), torch.tensor(1.0, dtype=torch.bfloat16)).float().item()
print(f"bf16 spacing at 0.02: {nxt - to_bf16(w):.3e}  (2^-13 = {2**-13:.3e})")
assert abs((nxt - to_bf16(w)) - 2**-13) < 1e-12
