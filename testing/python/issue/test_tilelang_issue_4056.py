"""Regression test for issue #4056.

Adding the GEMM accumulator to the row statistic of *another* fragment failed
layout inference with ``no available layout found``. The free fragment feeding
the reduction picked the layout of its producer loop, and ``ReduceOp`` only
ever inferred the destination from the source, so nothing could reconcile it
with the accumulation layout the consumer required.

``ReduceOp`` now inverts the reduction when the destination is already fixed,
so the source adopts the destination's thread distribution. These tests lower
the reported kernel (and the variants from the report) and check the result.
"""

import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm
from tvm import tirx

M = N = K = 64


def _kernel(threads: int, *, reduce: str = "sum", source: str = "other"):
    reduce_fn = T.reduce_sum if reduce == "sum" else T.reduce_max

    @T.prim_func
    def main(
        A: T.Tensor((M, K), "float16"),
        B: T.Tensor((K, N), "float16"),
        H: T.Tensor((M, N), "float16"),
        Out: T.Tensor((M, N), "float32"),
    ):
        with T.Kernel(1, threads=threads):
            As = T.alloc_shared((M, K), "float16")
            Bs = T.alloc_shared((K, N), "float16")
            acc = T.alloc_fragment((M, N), "float32")
            stat = T.alloc_fragment((M,), "float32")
            out = T.alloc_fragment((M, N), "float32")
            T.copy(A, As)
            T.copy(B, Bs)
            T.clear(acc)
            T.gemm(As, Bs, acc)
            if source == "other":
                wide = T.alloc_fragment((M, N), "float32")
                for i, j in T.Parallel(M, N):
                    wide[i, j] = T.cast(H[i, j], "float32")
                reduce_fn(wide, stat, dim=1, clear=True)
            else:
                reduce_fn(acc, stat, dim=1, clear=True)
            for i, j in T.Parallel(M, N):
                out[i, j] = stat[i] + acc[i, j]
            T.copy(out, Out)

    return main


def _layout_map(func, target):
    from tilelang.cuda.pipeline import CUDAPassPipelineBodyPrologue

    mod = tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    with target:
        mod = CUDAPassPipelineBodyPrologue(mod, target)
    maps = []

    def visit(node):
        if isinstance(node, tirx.SBlock):
            for key, value in node.annotations.items():
                if str(key) == "layout_map":
                    maps.append(value)

    tirx.stmt_functor.post_order_visit(mod["main"].body, visit)
    assert maps, "layout inference produced no layout map"
    return maps[0]


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("threads", [128, 256])
@pytest.mark.parametrize("reduce", ["sum", "max"])
def test_reduce_of_other_fragment_infers_source_layout(threads, reduce):
    """The reduction source adopts a layout the fixed destination accepts."""
    target = tvm.target.Target({"kind": "cuda", "arch": "sm_89"})
    layouts = _layout_map(_kernel(threads, reduce=reduce), target)
    buffers = {buffer.name for buffer in layouts}
    assert "wide" in buffers
    assert "stat" in buffers


@tilelang.testing.requires_cuda
def test_reduce_of_other_fragment_gemm_epilogue():
    """The reported epilogue produces the right numbers."""
    compiled = tilelang.jit(_kernel)(threads=128)

    torch.manual_seed(0)
    # Small integers are exact in float16, so the GEMM and row sums are exact.
    a = torch.randint(-2, 3, (M, K), device="cuda", dtype=torch.float16)
    b = torch.randint(-2, 3, (K, N), device="cuda", dtype=torch.float16)
    h = torch.randint(-2, 3, (M, N), device="cuda", dtype=torch.float16)
    expected = h.float().sum(dim=1, keepdim=True) + (a.float() @ b.float())

    out = torch.zeros((M, N), device="cuda", dtype=torch.float32)
    compiled(a, b, h, out)
    torch.cuda.synchronize()
    torch.testing.assert_close(out, expected)


if __name__ == "__main__":
    tilelang.testing.main()
