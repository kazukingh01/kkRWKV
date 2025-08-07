import gc, importlib
from types import SimpleNamespace
import torch
import torch.nn as nn
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning.strategies import DeepSpeedStrategy
if importlib.util.find_spec('deepspeed'):
    import deepspeed
    from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam

# local modules
from .block import Block, CustomEmbLayer, CustomHeadLayer


class RWKV(pl.LightningModule):
    def __init__(
        self, n_features: int, n_symbols: int, embd_dim: int=512, n_layers: int=6, seq_len: int=512, 
        num_classes: int=5, weight_decay: float=0.0, lr_init: float=6e-4, betas: tuple[float, float]=(0.9, 0.99),
        adam_eps: float=1e-18, mode_float: str="fp16", is_cpu: bool=False
    ):
        super().__init__()
        assert isinstance(n_features,  int) and n_features  > 0
        assert isinstance(embd_dim,    int) and embd_dim    > 0
        assert isinstance(n_layers,    int) and n_layers    > 0
        assert isinstance(seq_len,     int) and seq_len     > 0
        assert isinstance(n_symbols,   int) and n_symbols   > 0
        assert isinstance(num_classes, int) and num_classes > 1
        assert isinstance(weight_decay, (int, float)) and weight_decay >= 0.0
        assert isinstance(lr_init,   float) and lr_init > 0.0
        assert isinstance(betas,     tuple) and len(betas) == 2
        for x in betas: assert isinstance(x, float) and 0.0 <= x <= 1.0
        assert isinstance(adam_eps,  float) and adam_eps > 0.0
        assert isinstance(mode_float,  str) and mode_float in ["fp16", "bf16", "fp32"]
        assert isinstance(is_cpu, bool)
        assert embd_dim % 32 == 0
        assert seq_len % 32 == 0
        ## args
        args              = SimpleNamespace()
        args.n_embd       = embd_dim
        args.my_testing   = "x070"
        args.dim_att      = embd_dim
        args.head_size    = 64
        args.n_layer      = n_layers
        ## model
        self.emb          = CustomEmbLayer(n_features, seq_len, embd_dim)
        self.blocks       = nn.ModuleList([Block(args, i) for i in range(n_layers)])
        self.ln_out       = nn.LayerNorm(embd_dim)
        self.head         = CustomHeadLayer(args.n_embd, n_symbols, num_classes=num_classes)
        ## model config
        self.seq_len      = seq_len
        self.n_layer      = n_layers
        self.embd_dim     = embd_dim
        ## optimizer config
        self.weight_decay = weight_decay
        self.lr_init      = lr_init
        self.betas        = betas
        self.adam_eps     = adam_eps
        ## others
        self.is_cpu       = is_cpu
        self.mode_float   = mode_float

    def configure_optimizers(self):
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
            if self.deepspeed_offload:
                return DeepSpeedCPUAdam(optim_groups, lr=self.lr_init, betas=self.betas, eps=self.adam_eps, bias_correction=True, adamw_mode=True, amsgrad=False)
            return FusedAdam(optim_groups, lr=self.lr_init, betas=self.betas, eps=self.adam_eps, bias_correction=True, adam_w_mode=True, amsgrad=False)
        else:
            if self.deepspeed_offload:
                return DeepSpeedCPUAdam(optim_groups, lr=self.lr_init, betas=self.betas, eps=self.adam_eps, bias_correction=True, adamw_mode=False, weight_decay=0, amsgrad=False)
            return FusedAdam(optim_groups, lr=self.lr_init, betas=self.betas, eps=self.adam_eps, bias_correction=True, adam_w_mode=False, weight_decay=0, amsgrad=False)

    @property
    def deepspeed_offload(self) -> bool:
        strategy = self.trainer.strategy
        if isinstance(strategy, DeepSpeedStrategy):
            cfg = strategy.config["zero_optimization"]
            return cfg.get("offload_optimizer") or cfg.get("offload_param")
        return False

    def forward(self, idx):
        B, T = idx.size()
        assert T <= self.seq_len, "Cannot forward, model seq_len is exhausted."

        x = self.emb(idx)

        v_first = torch.empty_like(x)
        for block in self.blocks:
            x, v_first = block(x, v_first)

        x = self.ln_out(x)
        x = self.head(x)
        return x

    def training_step(self, batch, _):
        data, targets = batch
        logits = self(data)  # [batch_size, num_classes]
        _, n_task = targets.shape
        n_label = logits.shape[-1] // n_task
        total_loss = 0
        for task_idx in range(n_task):
            task_target = targets[:, task_idx]  # [batch_size]
            task_logit  = logits[:, task_idx * n_label:(task_idx + 1) * n_label]  # [batch_size, num_classes]
            task_loss   = F.cross_entropy(task_logit, task_target)
            total_loss += task_loss
        return total_loss / n_task

    def training_step_end(self, batch_parts):
        all = self.all_gather(batch_parts)
        if self.trainer.is_global_zero:
            self.trainer.my_loss_all = all

    def generate_init_weight(self):
        print(
            f"""
############################################################################
#
# Init model weight (slow for large models)...
#
############################################################################
"""
        )
        m = {}
        n_params = 0
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

                print(f" [scale {scale}]")

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

            m[n] = m[n].cpu()
            if self.mode_float == "fp16":
                m[n] = m[n].half()
            elif self.mode_float == "bf16":
                m[n] = m[n].bfloat16()
            n_params += m[n].numel()

        print('model params', n_params)
        gc.collect()
        torch.cuda.empty_cache()
        return m