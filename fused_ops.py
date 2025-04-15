from typing import Any, Dict, List
import torch
from auto_diff import *

class MatMulLayerNormOp(Op):
    """Fused matrix multiplication and layer normalization operation."""

    def __call__(
        self, 
        node_A: Node, 
        node_B: Node, 
        normalized_shape: List[int], 
        eps: float = 1e-5
    ) -> Node:
        """
        Args:
            node_A: The first input node.
            node_B: The second input node.
            normalized_shape: The shape of the normalization axes.
            eps: The epsilon value to avoid division by zero.
        """
        return Node(
            inputs=[node_A, node_B],
            op=self,
            attrs={
                "normalized_shape": normalized_shape,
                "eps": eps
            },
            name=f"MatMulLayerNorm({node_A.name}@{node_B.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the fused matmul and layer normalization result."""
        assert len(input_values) == 2
        res = torch.matmul(input_values[0], input_values[1])

        idx = list(range(len(node.normalized_shape) * -1, 0))
        mean = torch.mean(res, dim=idx, keepdim=True).expand_as(res)
        var = torch.var(res, dim=idx, keepdim=True, correction=0).expand_as(res)
        return (res - mean) / torch.sqrt(var + node.eps)

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of fused node, return partial adjoints to each input."""
        res = matmul(node.inputs[0], node.inputs[1])
        
        idx = list(range(len(node.normalized_shape) * -1, 0))
        m = mean(res, normalized_shape=node.normalized_shape, keepdim=True)
        s = std(res, normalized_shape=node.normalized_shape, keepdim=True, eps=node.eps, correction=0)
        N = 1
        for i in node.normalized_shape:
            N *= i

        dvar = sum_op(output_grad * (res - m) * -0.5 * power(s, -3) , dim=idx, keepdim=True)
        dmean = sum_op(output_grad * -1 / s, dim=idx, keepdim=True) + dvar * mean(-2 * (res - m), normalized_shape=node.normalized_shape, keepdim=True) / N
        dvar = expand_as(dvar, res, idx)
        dmean = expand_as(dmean, res, idx)
        grad = output_grad / expand_as(s, output_grad, idx) + dvar * 2 * (res - expand_as(m, res, idx)) / N + dmean / N
        return [matmul(grad, transpose(node.inputs[1], -2, -1)), matmul(transpose(node.inputs[0], -2, -1), grad)]


class MatMulSoftmaxOp(Op):
    """Fused matrix multiplication and softmax operation."""

    def __call__(
        self, 
        node_A: Node, 
        node_B: Node, 
        dim: int = -1
    ) -> Node:
        return Node(
            inputs=[node_A, node_B],
            op=self,
            attrs={
                "dim": dim
            },
            name=f"MatMulSoftmax({node_A.name}@{node_B.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the fused matmul and softmax result."""
        assert len(input_values) == 2
        res = torch.matmul(input_values[0], input_values[1])
        maxn = torch.max(res, dim=node.dim, keepdim=True).values
        res = res - maxn

        exp = torch.exp(res)
        sum = torch.sum(exp, dim=node.dim, keepdim=True).expand_as(exp)
        return exp / sum

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of fused node, return partial adjoints to each input."""
        # First compute the forward pass result we need for softmax gradient
        grad = node * (sub(output_grad, expand_as(sum_op(node * output_grad, dim=node.dim, keepdim=True), node, node.dim)))
        return [matmul(grad, transpose(node.inputs[1], -2, -1)), matmul(transpose(node.inputs[0], -2, -1), grad)]

# Create global instances of the fused ops
matmul_layernorm = MatMulLayerNormOp()
matmul_softmax = MatMulSoftmaxOp()