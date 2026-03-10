import triton
import triton.language as tl

@triton.jit
def tl_fp32_to_tf32(x):
    return tl.inline_asm_elementwise("cvt.rna.tf32.f32 $0, $1;", "=r, r", [x], dtype=tl.float32, is_pure=True, pack=1)