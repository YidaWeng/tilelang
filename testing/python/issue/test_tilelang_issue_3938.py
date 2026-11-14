"""Regression test for #3938: a conditional double-buffered K loop.

The loop alternates between two shared tiles with ``if k % 2 == 0: ... else:
...``.  ``ThreadSync`` used to compare the guard of the next iteration under the
current iteration variable, so the two arms looked mutually exclusive and the
loop-carried write-after-read / read-after-write hazards between consecutive
iterations were dropped.  With 256 threads the missing barrier made almost every
launch of the float32 (tf32) kernel return wrong values.

The mechanism is covered without a GPU in
``testing/python/transform/test_tilelang_transform_thread_sync.py``; this file
reproduces the user-visible failure from the report.
"""

import torch

import tilelang
import tilelang.language as T
import tilelang.testing

M = N = 64
K = 256
BK = 32
THREADS = 256


@tilelang.testing.requires_cuda
def test_conditional_double_buffer_gemm_is_correct():
    kt = K // BK

    @tilelang.jit
    def kernel():
        @T.prim_func
        def main(
            A: T.Tensor((M, K), "float32"),
            B: T.Tensor((K, N), "float32"),
            C: T.Tensor((M, N), "float32"),
        ):
            with T.Kernel(1, threads=THREADS):
                Af = T.alloc_fragment((M, BK), "float32")
                B0 = T.alloc_shared((BK, N), "float32")
                B1 = T.alloc_shared((BK, N), "float32")
                Cl = T.alloc_fragment((M, N), "float32")
                T.clear(Cl)
                T.copy(B[0, 0], B0)
                for k in T.serial(kt):
                    if k % 2 == 0:
                        if k + 1 < kt:
                            T.copy(B[(k + 1) * BK, 0], B1)
                        T.copy(A[0, k * BK], Af)
                        T.gemm(Af, B0, Cl)
                    else:
                        if k + 1 < kt:
                            T.copy(B[(k + 1) * BK, 0], B0)
                        T.copy(A[0, k * BK], Af)
                        T.gemm(Af, B1)
                T.copy(Cl, C)

        return main

    compiled = kernel()
    # Small integers are exact in tf32 as well, so the reference is exact.
    a = torch.randint(-2, 3, (M, K), device="cuda", dtype=torch.float32)
    b = torch.randint(-2, 3, (K, N), device="cuda", dtype=torch.float32)
    expected = a @ b
    # The race was observed on 9-10 of 10 launches, so a single run could pass.
    for _ in range(10):
        c = torch.zeros((M, N), device="cuda", dtype=torch.float32)
        compiled(a, b, c)
        torch.cuda.synchronize()
        torch.testing.assert_close(c, expected)


if __name__ == "__main__":
    tilelang.testing.main()
