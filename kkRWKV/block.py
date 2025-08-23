########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

# see: https://github.com/BlinkDL/RWKV-LM/blob/20d3a1f4eefa7c0f9762d8b94c2f6ee1eac1b27d/RWKV-v7/train_temp/src/model.py

import os, math
import torch
import torch.nn as nn
from torch.nn import functional as F

# PyTorchのテンソル表示オプションを設定
torch.set_printoptions(threshold=1000, linewidth=80, precision=4, profile="default")


def __nop(ob):
    return ob


MyModule = nn.Module
MyFunction = __nop

if os.environ.get("RWKV_JIT_ON") == "1":
    MyModule   = torch.jit.ScriptModule
    MyFunction = torch.jit.script_method


########################################################################################################
# CUDA Kernel
########################################################################################################

from torch.utils.cpp_extension import load


HEAD_SIZE = int(os.environ.get("RWKV_HEAD_SIZE", "-1"))

if os.environ.get("RWKV_TRAIN") == "1":
    assert HEAD_SIZE == 64 # Fix !!
    CHUNK_LEN = 16
    flags = ['-res-usage', f'-D_C_={HEAD_SIZE}', f"-D_CHUNK_LEN_={CHUNK_LEN}", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization"]
    _current_dir = os.path.dirname(os.path.abspath(__file__))
    cuda_sources = [
        os.path.join(_current_dir, 'cuda', 'wkv7_cuda.cu'),
        os.path.join(_current_dir, 'cuda', 'wkv7_op.cpp')
    ]
    load(name="wind_backstepping", sources=cuda_sources, is_python_module=False, verbose=True, extra_cuda_cflags=flags)

    class WindBackstepping(torch.autograd.Function):
        @staticmethod
        def forward(ctx, w,q,k,v,z,b):
            B,T,H,C = w.shape
            assert T%CHUNK_LEN == 0 # if T%CHUNK_LEN != 0: pad your input to T%CHUNK_LEN == 0, or change CHUNK_LEN (will be slower)
            assert all(i.dtype==torch.bfloat16 for i in [w,q,k,v,z,b])
            assert all(i.is_contiguous() for i in [w,q,k,v,z,b])
            y = torch.empty_like(v)
            s = torch.empty(B,H,T//CHUNK_LEN,C,C, dtype=torch.float32,device=w.device)
            sa = torch.empty(B,T,H,C, dtype=torch.float32,device=w.device)
            torch.ops.wind_backstepping.forward(w,q,k,v,z,b, y,s,sa)
            ctx.save_for_backward(w,q,k,v,z,b,s,sa)
            return y
        @staticmethod
        def backward(ctx, dy):
            assert all(i.dtype==torch.bfloat16 for i in [dy])
            assert all(i.is_contiguous() for i in [dy])
            w,q,k,v,z,b,s,sa = ctx.saved_tensors
            dw,dq,dk,dv,dz,db = [torch.empty_like(x) for x in [w,q,k,v,z,b]]
            torch.ops.wind_backstepping.backward(w,q,k,v,z,b, dy,s,sa, dw,dq,dk,dv,dz,db)
            return dw,dq,dk,dv,dz,db

    def RUN_CUDA_RWKV7g(q,w,k,v,a,b):
        B,T,HC = q.shape
        q,w,k,v,a,b = [i.view(B,T,HC//64,64) for i in [q,w,k,v,a,b]]
        return WindBackstepping.apply(w,q,k,v,a,b).view(B,T,HC)

def RWKV7_OP(r, w, k, v, a, b):
    B, T, C = r.size()
    H = C // 64 # HEAD SIZE
    N = 64  # HEAD SIZE
    r = r.view(B, T, H, N).float()
    k = k.view(B, T, H, N).float()
    v = v.view(B, T, H, N).float()
    a = a.view(B, T, H, N).float()
    b = b.view(B, T, H, N).float()
    w = torch.exp(-torch.exp(w.view(B, T, H, N).float()))
    out = torch.zeros((B, T, H, N), device=r.device, dtype=torch.float)
    state = torch.zeros((B, H, N, N), device=r.device, dtype=torch.float)

    for t in range(T):
        kk = k[:, t, :].view(B, H, 1, N)
        rr = r[:, t, :].view(B, H, N, 1)
        vv = v[:, t, :].view(B, H, N, 1)
        aa = a[:, t, :].view(B, H, N, 1)
        bb = b[:, t, :].view(B, H, 1, N)
        state = state * w[: , t, :, None, :] + state @ aa @ bb + vv @ kk
        out[:, t, :] = (state @ rr).view(B, H, N)

        # another method using einsum
        #
        # kk = k[:, t, :]
        # rr = r[:, t, :]
        # vv = v[:, t, :]
        # aa = a[:, t, :]
        # bb = b[:, t, :]
        # sab = torch.einsum('bhik,bhk,bhj->bhij', state, aa, bb)
        # state = state * w[: , t, :, None, :] + sab + torch.einsum('bhj,bhi->bhij', kk, vv)
        # out[:, t, :] = torch.einsum('bhj,bhij->bhi', rr, state)

    return out.view(B, T, C)

if os.environ.get("RWKV_TRAIN") == "1":
    RUN_RWKV7 = RUN_CUDA_RWKV7g
else:
    RUN_RWKV7 = RWKV7_OP

class RWKV_Tmix_x070(MyModule):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        self.head_size = args.head_size
        self.n_head = args.dim_att // self.head_size
        assert args.dim_att % self.n_head == 0
        H = self.n_head
        N = self.head_size
        C = args.n_embd

        with torch.no_grad():
            ratio_0_to_1 = layer_id / (args.n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, C)
            for i in range(C):
                ddd[0, 0, i] = i / C

            self.x_r = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))
            self.x_w = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
            self.x_k = nn.Parameter(1.0 - torch.pow(ddd, 0.7 * ratio_1_to_almost0))
            self.x_v = nn.Parameter(1.0 - torch.pow(ddd, 0.7 * ratio_1_to_almost0))
            self.x_a = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
            self.x_g = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))

            def ortho_init(x, scale):
                with torch.no_grad():
                    shape = x.shape
                    if len(shape) == 2:
                        gain = math.sqrt(shape[0] / shape[1]) if shape[0] > shape[1] else 1
                        nn.init.orthogonal_(x, gain=gain * scale)
                    elif len(shape) == 3:
                        gain = math.sqrt(shape[1] / shape[2]) if shape[1] > shape[2] else 1
                        for i in range(shape[0]):
                            nn.init.orthogonal_(x[i], gain=gain * scale)
                    else:
                        assert False
                    return x

            www = torch.zeros(C)
            zigzag = torch.zeros(C)
            linear = torch.zeros(C)
            for n in range(C):
                linear[n] = n / (C-1) - 0.5
                zigzag[n] = ((n % N) - ((N-1) / 2)) / ((N-1) / 2)
                zigzag[n] = zigzag[n] * abs(zigzag[n])
                www[n] = -6 + 6 * (n / (C - 1)) ** (1 + 1 * ratio_0_to_1 ** 0.3)

            D_DECAY_LORA = max(32, int(round(  (2.5*(C**0.5))  /32)*32)) # suggestion
            self.w1 = nn.Parameter(torch.zeros(C, D_DECAY_LORA))
            self.w2 = nn.Parameter(ortho_init(torch.zeros(D_DECAY_LORA, C), 0.1))
            self.w0 = nn.Parameter(www.reshape(1,1,C) + 0.5 + zigzag*2.5) # !!! 0.5 comes from F.softplus !!!

            D_AAA_LORA = max(32, int(round(  (2.5*(C**0.5))  /32)*32)) # suggestion
            self.a1 = nn.Parameter(torch.zeros(C, D_AAA_LORA))
            self.a2 = nn.Parameter(ortho_init(torch.zeros(D_AAA_LORA, C), 0.1))
            self.a0 = nn.Parameter(torch.zeros(1,1,C)-0.19 + zigzag*0.3 + linear*0.4)

            D_MV_LORA = max(32, int(round(  (1.7*(C**0.5))  /32)*32)) # suggestion
            self.v1 = nn.Parameter(torch.zeros(C, D_MV_LORA))
            self.v2 = nn.Parameter(ortho_init(torch.zeros(D_MV_LORA, C), 0.1))
            self.v0 = nn.Parameter(torch.zeros(1,1,C)+0.73 - linear*0.4)

            # Note: for some data, you can reduce D_GATE_LORA or even remove this gate
            D_GATE_LORA = max(32, int(round(  (5*(C**0.5))  /32)*32)) # suggestion
            self.g1 = nn.Parameter(torch.zeros(C, D_GATE_LORA))
            self.g2 = nn.Parameter(ortho_init(torch.zeros(D_GATE_LORA, C), 0.1))

            self.k_k = nn.Parameter(torch.zeros(1,1,C)+0.71 - linear*0.1)
            self.k_a = nn.Parameter(torch.zeros(1,1,C)+1.02)
            self.r_k = nn.Parameter(torch.zeros(H,N)-0.04)

            self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
            self.receptance = nn.Linear(C, C, bias=False)
            self.key = nn.Linear(C, C, bias=False)
            self.value = nn.Linear(C, C, bias=False)
            self.output = nn.Linear(C, C, bias=False)
            self.ln_x = nn.GroupNorm(H, C, eps=64e-5) # !!! notice eps value !!!

            self.receptance.weight.data.uniform_(-0.5/(C**0.5), 0.5/(C**0.5))
            self.key.weight.data.uniform_(-0.05/(C**0.5), 0.05/(C**0.5))
            self.value.weight.data.uniform_(-0.5/(C**0.5), 0.5/(C**0.5))
            self.output.weight.data.zero_()

    @MyFunction
    def forward(self, x, v_first):
        B, T, C = x.size()
        H = self.n_head
        xx = self.time_shift(x) - x

        xr = x + xx * self.x_r
        xw = x + xx * self.x_w
        xk = x + xx * self.x_k
        xv = x + xx * self.x_v
        xa = x + xx * self.x_a
        xg = x + xx * self.x_g

        r = self.receptance(xr)
        w = -F.softplus(-(self.w0 + torch.tanh(xw @ self.w1) @ self.w2)) - 0.5 # soft-clamp to (-inf, -0.5)
        k = self.key(xk)
        v = self.value(xv)
        if self.layer_id == 0:
            v_first = v # store the v of the first layer
        else:
            v = v + (v_first - v) * torch.sigmoid(self.v0 + (xv @ self.v1) @ self.v2) # add value residual
        a = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2) # a is "in-context learning rate"
        g = torch.sigmoid(xg @ self.g1) @ self.g2

        kk = k * self.k_k
        kk = F.normalize(kk.view(B,T,H,-1), dim=-1, p=2.0).view(B,T,C)
        k = k * (1 + (a-1) * self.k_a)

        x = RUN_RWKV7(r, w, k, v, -kk, kk*a)
        x = self.ln_x(x.view(B * T, C)).view(B, T, C)

        x = x + ((r.view(B,T,H,-1)*k.view(B,T,H,-1)*self.r_k).sum(dim=-1, keepdim=True) * v.view(B,T,H,-1)).view(B,T,C)
        x = self.output(x * g)
        return x, v_first
    

class RWKV_CMix_x070(MyModule):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        with torch.no_grad():
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                ddd[0, 0, i] = i / args.n_embd
            self.x_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0**4))

        self.key = nn.Linear(args.n_embd, args.n_embd * 4, bias=False)
        self.value = nn.Linear(args.n_embd * 4, args.n_embd, bias=False)

        self.key.weight.data.uniform_(-0.5/(args.n_embd**0.5), 0.5/(args.n_embd**0.5))
        self.value.weight.data.zero_()

    @MyFunction
    def forward(self, x):
        xx = self.time_shift(x) - x
        
        k = x + xx * self.x_k
        k = torch.relu(self.key(k)) ** 2

        return self.value(k)


class Block(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(args.n_embd)
        self.ln2 = nn.LayerNorm(args.n_embd)

        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(args.n_embd)

        self.att = RWKV_Tmix_x070(args, layer_id)
        self.ffn = RWKV_CMix_x070(args, layer_id)
        
    def forward(self, x, v_first):
        if self.layer_id == 0:
            x = self.ln0(x)

        x_attn, v_first = self.att(self.ln1(x), v_first)
        x = x + x_attn

        x = x + self.ffn(self.ln2(x))
        return x, v_first


class L2Wrap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, loss, y):
        ctx.save_for_backward(y)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        y = ctx.saved_tensors[0]
        # to encourage the logits to be close to 0
        factor = 1e-4 / (y.shape[0] * y.shape[1])
        maxx, ids = torch.max(y, -1, keepdim=True)
        gy = torch.zeros_like(y)
        gy.scatter_(-1, ids, maxx * factor)
        return (grad_output, gy)


class CustomEmbLayer(nn.Module):
    def __init__(
        self, dim_ln: int, dim_emb1_in: int, dim_emb1_out: int, dim_emb2_in: int, dim_emb2_out: int, 
        dim_other: int, seq_len: int, output_dim: int
    ):
        super().__init__()
        assert isinstance(dim_ln,     int) and dim_ln     > 0
        assert isinstance(dim_emb1_in, int) and dim_emb1_in > 0
        assert isinstance(dim_emb1_out, int) and dim_emb1_out > 0
        assert isinstance(dim_emb2_in, int) and dim_emb2_in > 0
        assert isinstance(dim_emb2_out, int) and dim_emb2_out > 0
        assert isinstance(dim_other,  int) and dim_other  > 0
        assert isinstance(seq_len,    int) and seq_len    > 0
        assert isinstance(output_dim, int) and output_dim > 0
        self.dim_ln       = dim_ln
        self.dim_emb1_in  = dim_emb1_in
        self.dim_emb1_out = dim_emb1_out
        self.dim_emb2_in  = dim_emb2_in
        self.dim_emb2_out = dim_emb2_out
        self.dim_other    = dim_other
        self.output_dim   = output_dim
        self.ln           = nn.LayerNorm(seq_len, elementwise_affine=True)
        self.bn           = nn.BatchNorm1d(num_features=dim_ln)
        self.emb1         = nn.Embedding(num_embeddings=dim_emb1_in, embedding_dim=dim_emb1_out)
        self.emb2         = nn.Embedding(num_embeddings=dim_emb2_in, embedding_dim=dim_emb2_out)
        self.linear       = nn.Linear(
            2 * self.dim_ln + self.dim_emb1_out + self.dim_emb2_out + self.dim_other, self.output_dim, bias=True
        )

    def forward(self, x: torch.Tensor):
        assert x.ndim == 3
        in_ln    = x[:, :, :self.dim_ln]
        in_emb1  = x[:, :,  self.dim_ln    ].to(torch.long)
        in_emb2  = x[:, :,  self.dim_ln + 1].to(torch.long)
        in_other = x[:, :,  self.dim_ln + 2:]
        out_ln   = self.ln(in_ln.permute(0, 2, 1)).permute(0, 2, 1)
        out_bn   = self.bn(in_ln.permute(0, 2, 1)).permute(0, 2, 1)
        out_emb1 = self.emb1(in_emb1)
        out_emb2 = self.emb2(in_emb2)
        output   = self.linear(torch.cat([out_ln, out_bn, out_emb1, out_emb2, in_other], dim=-1))
        return output


class CustomHeadLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        assert isinstance(input_dim,   int) and input_dim   > 0
        assert isinstance(output_dim,  int) and output_dim  > 0
        self.fnn = nn.Linear(input_dim, output_dim, bias=True)

    def forward(self, x):
        x = self.fnn(x)
        x = x.sum(dim=1)
        return x


def time_mixing__(layer_id:int, H:int, N:int, x, x_prev, v_first, state, x_r, x_w, x_k, x_v, x_a, x_g, w0, w1, w2, a0, a1, a2, v0, v1, v2, g1, g2, k_k, k_a, r_k, kw, vw, rw, ow, ln_w, ln_b):
    xx = x_prev - x
    xr, xw, xk, xv, xa, xg = x+xx*x_r, x+xx*x_w, x+xx*x_k, x+xx*x_v, x+xx*x_a, x+xx*x_g
    r = rw @ xr
    _w = torch.tanh(xw @ w1) @ w2
    k = kw @ xk
    v = vw @ xv
    a = torch.sigmoid(a0 + (xa @ a1) @ a2)
    g = torch.sigmoid(xg @ g1) @ g2
    kk = k * k_k
    kk = torch.nn.functional.normalize(kk.view(H,N), dim=-1, p=2.0).view(-1)
    k = k * (1 + (a-1) * k_a)
    if layer_id == 0:
        v_first = v
    else:
        v = v + (v_first - v) * torch.sigmoid(v0 + (xv @ v1) @ v2)

    w = -torch.nn.functional.softplus(-(w0 + _w)) - 0.5
    assert w.dtype == torch.float
    w = torch.exp(-torch.exp(w))

    # rwkv-7 kernel
    vk = v.view(H,N,1) @ k.view(H,1,N)
    ab = (-kk).view(H,N,1) @ (kk*a).view(H,1,N)
    state = state * w.view(H,1,N) + state @ ab.float() + vk.float()
    out = state.to(dtype=x.dtype) @ r.view(H,N,1)
    out = out.float()
    # torch.nn.functional.group_norm 's calculation is not same as compiled "self.ln_x = nn.GroupNorm(H, C, eps=64e-5)".
    # It's not because of nn.GroupNorm. nn.functional.group_norm and nn.GroupNorm result same here.
    out = torch.nn.functional.group_norm(out.view(1,H*N), num_groups=H, weight=ln_w, bias=ln_b, eps = 64e-5).view(H*N)
    out = out.to(x.dtype)
    out = out + ((r * k * r_k).view(H,N).sum(dim=-1, keepdim=True) * v.view(H,N)).view(H*N)
    return ow @ (out * g), x, state, v_first
try:
    time_mixing = torch.compile(time_mixing__, mode="max-autotune", fullgraph=True, dynamic=False)
except:
    time_mixing = torch.jit.script(time_mixing__)

########################################################################################################

def channel_mixing__(x, x_prev, x_k, kw, vw):
    xx = x_prev - x
    k = x + xx * x_k
    k = torch.relu(kw @ k) ** 2
    return vw @ k, x
try:
    channel_mixing = torch.compile(channel_mixing__, mode="max-autotune", fullgraph=True, dynamic=False)
except:
    channel_mixing = torch.jit.script(channel_mixing__)
