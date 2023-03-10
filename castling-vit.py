import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# if use Q @ K, FLOPs caclulation could be wrong
class MatMul(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, a, b):
        out = a @ b
        return out

class LinAngularAttention(nn.Module):
    def __init__(
        self,
        in_channels,
        num_heads=8,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
        res_kernel_size=9,
        sparse_reg=False,
    ):
        super().__init__()
        assert in_channels % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = in_channels // num_heads
        self.scale = head_dim**-0.5
        self.sparse_reg = sparse_reg

        self.qkv = nn.Linear(in_channels, in_channels * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(in_channels, in_channels)
        self.proj_drop = nn.Dropout(proj_drop)

        self.kq_matmul = MatMul()
        self.kqv_matmul = MatMul()
        if self.sparse_reg:
            self.qk_matmul = MatMul()
            self.sv_matmul = MatMul()

        self.dconv = nn.Conv2d(
            in_channels=self.num_heads,
            out_channels=self.num_heads,
            kernel_size=(res_kernel_size, 1),
            padding=(res_kernel_size // 2, 0),
            bias=False,
            groups=self.num_heads,
        )

    def forward(self, x):
        N, L, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(N, L, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        if self.sparse_reg:
            attn = self.qk_matmul(q * self.scale, k.transpose(-2, -1))
            attn = attn.softmax(dim=-1)
            mask = attn > 0.02 # note that the threshold could be different; adapt to your codebases.
            sparse = mask * attn

        q = q / q.norm(dim=-1, keepdim=True)
        k = k / k.norm(dim=-1, keepdim=True)
        dconv_v = self.dconv(v)

        attn = self.kq_matmul(k.transpose(-2, -1), v)

        if self.sparse_reg:
            x = (
                self.sv_matmul(sparse, v)
                + 0.5 * v
                + 1.0 / math.pi * self.kqv_matmul(q, attn)
            )
        else:
            x = 0.5 * v + 1.0 / math.pi * self.kqv_matmul(q, attn)
        x = x / x.norm(dim=-1, keepdim=True)
        x += dconv_v
        x = x.transpose(1, 2).reshape(N, L, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input = torch.randn(1, 196, 256).to(device)

    # linear angular attention with DWConv
    linear_angular_attention = LinAngularAttention(in_channels=256, num_heads=8, qkv_bias=False, sparse_reg=False).to(device)
    output = linear_angular_attention(input)
    print(output.shape)

    # linear angular attention with DWConv + SparseAttn
    linear_angular_attention = LinAngularAttention(in_channels=256, num_heads=8, qkv_bias=False, sparse_reg=True).to(device)
    output = linear_angular_attention(input)
    print(output.shape)