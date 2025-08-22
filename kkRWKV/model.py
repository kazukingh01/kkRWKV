import gc, os, importlib
from typing import Self
from types import SimpleNamespace
import torch
import torch.nn as nn
from torch.nn import functional as F
import lightning as L
from deepspeed.ops.adam import FusedAdam
from kklogger import set_logger
LOGGER = set_logger(__name__)

# local modules
import kkRWKV
from kkRWKV.metrics import calc_cross_entropy, calc_accuracy
from kkRWKV.block import CustomEmbLayer, CustomHeadLayer, time_mixing, channel_mixing

# env
torch.set_printoptions(threshold=1000, linewidth=80, precision=4, profile="default")
HEAD_SIZE = 64
assert HEAD_SIZE == 64 # Fix !!


class RWKV(nn.Module):
    def __init__(
        self, n_features: int, n_symbols: int, embd_dim: int=512, n_layers: int=6, seq_len: int=512, 
        num_classes: int=5, mode_float: str="bf16", is_cpu: bool=False, is_jit: bool=True
    ):
        LOGGER.info("START")
        super().__init__()
        assert isinstance(n_features,  int) and n_features  > 0
        assert isinstance(embd_dim,    int) and embd_dim    > 0
        assert isinstance(n_layers,    int) and n_layers    > 0
        assert isinstance(seq_len,     int) and seq_len     > 0
        assert isinstance(n_symbols,   int) and n_symbols   > 0
        assert isinstance(num_classes, int) and num_classes > 1
        assert isinstance(mode_float,  str) and mode_float in ["fp16", "bf16", "fp32"]
        assert isinstance(is_cpu, bool)
        assert isinstance(is_jit, bool)
        assert embd_dim % HEAD_SIZE == 0
        assert seq_len  % 32 == 0
        ## set environment variables
        os.environ["RWKV_HEAD_SIZE"] = f"{HEAD_SIZE}" # Fix !!
        os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9"
        os.environ["RWKV_TRAIN"]     = "1" if mode_float == "bf16" else "0"
        os.environ["RWKV_JIT_ON"]    = "1" if is_jit   else "0"
        importlib.reload(kkRWKV.block)
        from kkRWKV.block import Block
        ## args
        args              = SimpleNamespace()
        args.n_embd       = embd_dim
        args.dim_att      = embd_dim
        args.head_size    = HEAD_SIZE  # n_head = args.dim_att // self.head_size
        args.n_layer      = n_layers
        ## model
        self.emb          = CustomEmbLayer(n_features, seq_len, embd_dim)
        self.blocks       = nn.ModuleList([Block(args, i) for i in range(n_layers)])
        self.ln_out       = nn.LayerNorm(embd_dim)
        self.head         = CustomHeadLayer(embd_dim, n_symbols * num_classes)
        ## model config
        self.seq_len      = seq_len
        self.n_layer      = n_layers
        self.embd_dim     = embd_dim
        self.num_classes  = num_classes
        self.n_symbols    = n_symbols
        ## others
        self.is_cpu       = is_cpu
        self.mode_float   = mode_float
        self._dtype       = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "fp32": torch.float32
        }[mode_float]
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        if self.mode_float == "fp32":
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cuda.matmul.allow_tf32 = False
        else:
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.matmul.allow_tf32 = True
        LOGGER.info("END")

    def forward(self, x: torch.Tensor):
        output = self.emb(x)
        v_first = torch.empty_like(output)
        for block in self.blocks:
            output, v_first = block(output, v_first)
        output = self.ln_out(output)
        output = self.head(output)
        return output

    def generate_init_weight(self):
        LOGGER.info("START")
        m = {}
        n_params = 0
        with torch.no_grad():
            for n in self.state_dict():
                p = self.state_dict()[n]
                shape = p.shape

                s0 = str(shape[0]) if len(shape) > 0 else ""
                s1 = str(shape[1]) if len(shape) > 1 else ""
                s2 = str(shape[2]) if len(shape) > 2 else ""
                s3 = str(shape[3]) if len(shape) > 3 else ""
                print(f"{s0.ljust(5)} {s1.ljust(5)} {s2.ljust(5)} {s3.ljust(5)} {n}", end="")

                scale = 1.0
                if "ln_" in n or ".ln" in n or "time_" in n or "_mask" in n or "pos_emb" in n or '.mask.' in n or n.endswith('_w') or n.endswith('_w1') or n.endswith('_w2') or n.endswith('_bias') or (".weight" not in n):
                    if 'ln_x.weight' in n:
                        layer_scale = (1+int(n.split('.')[1])) / self.n_layer
                        m[n] = (p * 0.0) + (layer_scale ** 0.7)
                    else:
                        m[n] = p
                    print()
                elif n == "emb.linear.weight":
                    m[n] = p
                    scale = -1e-4
                    nn.init.uniform_(m[n], a=scale, b=-scale)
                    print(f" [scale {scale}]")
                elif n == "head.fnn.weight":
                    m[n] = p
                    scale = 0.5
                    nn.init.orthogonal_(m[n], gain=scale)
                    print(f" [scale {scale}]")
                else:
                    assert n.endswith('.weight') # should always be true

                    zero = [".att.output.", ".ffn.value.", ".ffn.receptance.", ".ffnPre.value.", ".ffnPre.receptance.", "head_q.", '.oo.', '.rr.']

                    for kk in zero:
                        if kk in n:
                            scale = 0

                    for kk in [".att.key."]:
                        if kk in n:
                            scale = 0.1
                    for kk in [".att.gate."]:
                        if kk in n:
                            scale = 0.1
                    if self.is_cpu:
                        m[n] = torch.empty((shape[0], shape[1]))
                    else:
                        m[n] = torch.empty((shape[0], shape[1]), device="cuda")

                    if scale == 0:
                        nn.init.zeros_(m[n])
                    elif scale < 0:
                        nn.init.uniform_(m[n], a=scale, b=-scale)
                    else:
                        nn.init.orthogonal_(m[n], gain=scale)
                    print(f" [scale {scale}]")
                m[n] = m[n].cpu()
                if self.mode_float == "fp16":
                    m[n] = m[n].half()
                elif self.mode_float == "bf16":
                    m[n] = m[n].bfloat16()
                n_params += m[n].numel()
        print('model params', n_params)
        gc.collect()
        self.load_state_dict(m, strict=True)
        LOGGER.info("END")
        return m
    
    @classmethod
    def load_from_model_path(cls, model_path: str, num_classes: int=5, mode_float: str="fp32", is_cpu: bool=False, is_jit: bool=True) -> Self:
        LOGGER.info("START")
        params    = torch.load(model_path, map_location="cpu")["state_dict"]
        embd_dim, n_features = params["emb.linear.weight"].shape
        seq_len   = params["emb.ln.weight"].shape[0]
        n_layers  = len(set([x.split(".")[1] for x in params.keys() if x.startswith("blocks.")]))
        out_dim   = params["head.fnn.weight"].shape[0]
        assert out_dim % num_classes == 0
        n_symbols = out_dim // num_classes
        model = cls(
            n_features, n_symbols, embd_dim=embd_dim, n_layers=n_layers, seq_len=seq_len,
            num_classes=num_classes, mode_float=mode_float, is_cpu=is_cpu, is_jit=is_jit
        )
        model.load_state_dict(params, strict=True)
        LOGGER.info("END")
        return model.to(model._dtype)
    
    def predict_proba(self, x: torch.Tensor):
        assert isinstance(x, torch.Tensor)
        assert x.ndim in [2,3]
        with torch.no_grad():
            if x.ndim == 2:
                out = x.unsqueeze(0)
            else:
                out = x
            out = out.to(self._dtype).to("cuda" if not self.is_cpu else "cpu")
            out = self.forward(out)
            for i in range(out.shape[-1] // self.num_classes):
                out[:, i * self.num_classes : (i + 1) * self.num_classes] = \
                    F.softmax(out[:, i * self.num_classes:(i + 1) * self.num_classes], dim=-1)
            if x.ndim == 2:
                out = out.squeeze(0)
            return out


class RWKV_FOR_TRAINING(RWKV, L.LightningModule):
    def __init__(
        self, n_features: int, n_symbols: int, embd_dim: int=512, n_layers: int=6, seq_len: int=512, 
        num_classes: int=5, weight_decay: float=0.0, lr_init: float=6e-4, betas: tuple[float, float]=(0.9, 0.99),
        adam_eps: float=1e-18, is_cpu: bool=False
    ):
        LOGGER.info("START")
        assert isinstance(weight_decay, (int, float)) and weight_decay >= 0.0
        assert isinstance(lr_init,   float) and lr_init > 0.0
        assert isinstance(betas,     tuple) and len(betas) == 2
        for x in betas: assert isinstance(x, float) and 0.0 <= x <= 1.0
        assert isinstance(adam_eps,  float) and adam_eps > 0.0
        super().__init__(
            n_features, n_symbols, embd_dim=embd_dim, n_layers=n_layers, seq_len=seq_len, 
            num_classes=num_classes, mode_float="bf16", is_cpu=is_cpu, is_jit=True
        )
        ## optimizer config
        self.weight_decay = weight_decay
        self.lr_init      = lr_init
        self.betas        = betas
        self.adam_eps     = adam_eps
        LOGGER.info("END")

    def configure_optimizers(self):
        LOGGER.info("START")
        lr_decay = set()
        lr_1x = set()
        lr_2x = set()
        for n, p in self.named_parameters():
            if ("att.w0" in n):
                lr_2x.add(n)
            elif (len(p.squeeze().shape) >= 2) and (self.weight_decay > 0) and (".weight" in n):
                lr_decay.add(n)
            else:
                lr_1x.add(n)

        lr_decay = sorted(list(lr_decay))
        lr_1x = sorted(list(lr_1x))
        lr_2x = sorted(list(lr_2x))

        if self.trainer.is_global_zero:
            print('decay', lr_decay, '\n')
            print('1x', lr_1x, '\n')
            print('2x', lr_2x, '\n')

        param_dict = {n: p for n, p in self.named_parameters()}
        
        optim_groups = [
            {"params": [param_dict[n] for n in lr_1x], "weight_decay": 0.0, "my_lr_scale": 1.0},
            {"params": [param_dict[n] for n in lr_2x], "weight_decay": 0.0, "my_lr_scale": 2.0},
        ]

        if self.weight_decay > 0:
            optim_groups += [{"params": [param_dict[n] for n in lr_decay], "weight_decay": self.weight_decay, "my_lr_scale": 1.0}]
            opt = FusedAdam(optim_groups, lr=self.lr_init, betas=self.betas, eps=self.adam_eps, bias_correction=True, adam_w_mode=True, amsgrad=False)
        else:
            opt = FusedAdam(optim_groups, lr=self.lr_init, betas=self.betas, eps=self.adam_eps, bias_correction=True, adam_w_mode=False, weight_decay=0, amsgrad=False)
        LOGGER.info("END")
        return opt

    def training_step(self, batch, _):
        data, targets = batch
        logits = self(data)  # [batch_size, num_classes]
        loss = calc_cross_entropy(logits, targets)
        self.log("train/loss", loss, prog_bar=True, on_step=True, sync_dist=True)
        total_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), float('inf'))
        self.log("train/grad_norm", total_norm, on_step=True)
        return loss

    def validation_step(self, batch, _):
        data, targets = batch
        logits = self(data)
        loss = calc_cross_entropy(logits, targets)
        acc  = calc_accuracy(logits, targets)
        self.log("val/loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("val/acc",  acc,  prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        return loss


class RWKV_FOR_INFERENCE(torch.jit.ScriptModule):
    def __init__(self, model_path: str, dtype: torch.dtype=torch.float32):
        LOGGER.info("START")
        super().__init__()
        self.dtype = dtype
        
        self.z = torch.load(model_path, map_location='cpu')["state_dict"]
        self.n_head, self.head_size = self.z['blocks.0.att.r_k'].shape
        self.embd_dim = self.z["emb.linear.weight"].shape[0]
        self.n_layer = max([int(x.split(".")[1]) for x in self.z.keys() if x.startswith("blocks")]) + 1

        # main blocks
        for k in list(self.z.keys()):
            if k.endswith('att.w0'): # convert to float
                self.z[k] = self.z[k].to(torch.float)
            elif k.endswith('att.ln_x.weight'): # convert to float
                self.z[k] = self.z[k].to(torch.float)
            elif k.endswith('att.ln_x.bias'): # convert to float
                self.z[k] = self.z[k].to(torch.float)
            else:
                self.z[k] = self.z[k].to(dtype=self.dtype)
            self.z[k] = self.z[k].squeeze()
            if k.endswith('att.r_k'): self.z[k] = self.z[k].flatten()
        
        # embedding layer
        _, n_features = self.z["emb.linear.weight"].shape
        seq_len  = self.z["emb.ln.weight"].shape[0]
        self.emb = CustomEmbLayer(n_features, seq_len, self.embd_dim).to(self.dtype)
        self.emb.load_state_dict({x.replace("emb.", ""): y for x, y in self.z.items() if x.startswith("emb.")}, strict=True)

        # head layer
        output_dim, _ = self.z["head.fnn.weight"].shape
        self.head     = CustomHeadLayer(self.embd_dim, output_dim).to(self.dtype)
        self.head.load_state_dict({x.replace("head.", ""): y for x, y in self.z.items() if x.startswith("head.")}, strict=True)

        # others
        self.z['blocks.0.att.v0'] = self.z['blocks.0.att.a0'] # actually ignored
        self.z['blocks.0.att.v1'] = self.z['blocks.0.att.a1'] # actually ignored
        self.z['blocks.0.att.v2'] = self.z['blocks.0.att.a2'] # actually ignored
        self.eval()
        LOGGER.info("END")

    def forward(self, x: torch.Tensor, seq_len: int=None):
        assert x.ndim == 2
        if seq_len is None: seq_len = x.shape[0]
        assert isinstance(seq_len, int) and seq_len > 0
        with torch.no_grad():
            state = [None for _ in range(self.n_layer * 3)]
            for i in range(self.n_layer): # state: 0=att_x_prev 1=att_kv 2=ffn_x_prev
                state[i*3+0] = torch.zeros(self.embd_dim, dtype=x.dtype, requires_grad=False)
                state[i*3+1] = torch.zeros((self.embd_dim // self.head_size, self.head_size, self.head_size), dtype=torch.float, requires_grad=False)
                state[i*3+2] = torch.zeros(self.embd_dim, dtype=x.dtype, requires_grad=False)
            x = self.emb(x.unsqueeze(0))[0]
            list_output = []
            for _x in x[:seq_len]:
                _output, state = self.forward_for_blocks(_x, state)
                list_output.append(_output.clone())
            x = F.layer_norm(torch.stack(list_output), (self.embd_dim,), weight=self.z['ln_out.weight'], bias=self.z['ln_out.bias'])
            x = self.head(x.unsqueeze(0))[0]
            return x

    def forward_for_blocks(self, x: torch.Tensor, state: list[torch.Tensor]):
        z = self.z
        v_first = torch.empty_like(x)
        x = F.layer_norm(x, (self.embd_dim,), weight=z['blocks.0.ln0.weight'], bias=z['blocks.0.ln0.bias'])
        for i in range(self.n_layer):
            bbb = f'blocks.{i}.'
            att = f'blocks.{i}.att.'
            ffn = f'blocks.{i}.ffn.'

            xx = F.layer_norm(x, (self.embd_dim,), weight=z[bbb+'ln1.weight'], bias=z[bbb+'ln1.bias'])

            xx, state[i*3+0], state[i*3+1], v_first = time_mixing(i, self.n_head, self.head_size, xx, state[i*3+0], v_first, state[i*3+1],
                z[att+'x_r'], z[att+'x_w'], z[att+'x_k'], z[att+'x_v'], z[att+'x_a'], z[att+'x_g'],
                z[att+'w0'], z[att+'w1'], z[att+'w2'], z[att+'a0'], z[att+'a1'], z[att+'a2'], z[att+'v0'], z[att+'v1'], z[att+'v2'],
                z[att+'g1'], z[att+'g2'], z[att+'k_k'], z[att+'k_a'], z[att+'r_k'],
                z[att+'key.weight'], z[att+'value.weight'], z[att+'receptance.weight'], z[att+'output.weight'],
                z[att+'ln_x.weight'], z[att+'ln_x.bias'])
            x = x + xx

            xx = F.layer_norm(x, (self.embd_dim,), weight=z[bbb+'ln2.weight'], bias=z[bbb+'ln2.bias'])

            xx, state[i*3+2] = channel_mixing(xx, state[i*3+2], z[ffn+'x_k'], z[ffn+'key.weight'], z[ffn+'value.weight'])
            x = x + xx

        return x, state
