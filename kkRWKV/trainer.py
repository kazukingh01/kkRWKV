import lightning as L
from lightning.pytorch.callbacks import Callback


class TrainCallback(Callback):
    def __init__(self, warmup_steps: int=-1, lr_init: float=6e-4, weight_decay: float=0.0, epoch_save: int=10, save_dir: str="./out"):
        assert isinstance(warmup_steps, int         ) and warmup_steps >= -1
        assert isinstance(lr_init,      float       ) and lr_init > 0.0
        assert isinstance(weight_decay, (float, int)) and weight_decay >= 0.0
        assert isinstance(epoch_save,   int         ) and epoch_save > 0
        assert isinstance(save_dir,     str         ) and save_dir != "" and save_dir.startswith("./")
        super().__init__()
        self.warmup_steps = warmup_steps
        self.lr_init      = lr_init
        self.weight_decay = float(weight_decay)
        self.epoch_save   = epoch_save
        self.save_dir     = save_dir

    def _init_logging_if_needed(self, trainer: L.Trainer, pl_module: L.LightningModule):
        if trainer.is_global_zero:
            # counters
            trainer.my_loss_sum = 0.0
            trainer.my_loss_count = 0
            # text log
            cfg = getattr(trainer.strategy, "config", None) # DeepSpeedStrategy 専用の属性
            if cfg is not None:
                trainer.log(f"{cfg}\n")

    def on_train_batch_start(self, trainer: L.Trainer, pl_module: L.LightningModule, batch, batch_idx: int) -> None:
        self._init_logging_if_needed(trainer, pl_module)
        lr = getattr(trainer, "my_lr", self.lr_init)
        if trainer.global_step < self.warmup_steps:
            warm_mult = (0.01 + 0.99 * trainer.global_step / max(1, self.warmup_steps))
            lr = lr * warm_mult
        trainer.my_lr = lr
        pl_module.log("train/lr", lr, on_step=True)
        pl_module.log("train/weight_decay", self.weight_decay, on_step=True)
        # print learning rate for each group
        if trainer.optimizers:
            opt = trainer.optimizers[0]
            for i, pg in enumerate(opt.param_groups):
                pl_module.log(f"train/lr_group_{i}", pg["lr"], on_step=True)

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if trainer.is_global_zero:
            if (trainer.current_epoch % self.epoch_save == 0) or (trainer.current_epoch == trainer.max_epochs - 1):
                import os
                os.makedirs(self.save_dir, exist_ok=True)
                save_path = f"{self.save_dir}/rwkv-epoch-{trainer.current_epoch}.pth"
                try:
                    trainer.save_checkpoint(save_path, weights_only=True)
                    print(f"Model saved: {save_path}")
                except Exception as e:
                    print(f"Error saving model: {e}")

