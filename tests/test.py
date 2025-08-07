import argparse, os
import torch
from pytorch_lightning import Trainer
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning import seed_everything

from kkRWKV.dataset import TimeSeriesDataset
from kkRWKV.trainer import train_callback
if "deepspeed" in args.strategy:
    import deepspeed

from kklogger import set_logger
LOGGER = set_logger(__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--train", action="store_true")
    args = parser.parse_args()
    LOGGER.info(f"{args}")

    if args.random_seed >= 0:
        print(f"########## WARNING: GLOBAL SEED {args.random_seed} THIS WILL AFFECT MULTIGPU SAMPLING ##########\n" * 3)
        seed_everything(args.random_seed)

    os.environ["RWKV_HEAD_SIZE"] = "64"
    if args.train:
        os.environ["RWKV_JIT_ON"] = "1"
        os.environ["RWKV_TRAIN"]  = "1"
        from kkRWKV.model import RWKV
        model = RWKV(n_features=10, n_symbols=10)
    else:
        model = None

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    if args.precision == "fp32":
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
    else:
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True

    trainer = Trainer.from_argparse_args(
        args,
        callbacks=[train_callback(args)],
    )
    model.generate_init_weight()
    if "deepspeed" in args.strategy:
        trainer.strategy.config["zero_optimization"]["allgather_bucket_size"] = args.ds_bucket_mb * 1000 * 1000
        trainer.strategy.config["zero_optimization"]["reduce_bucket_size"] = args.ds_bucket_mb * 1000 * 1000

    train_data = TimeSeriesDataset("getdata/data.csv", seq_len=128)
    data_loader = DataLoader(train_data, shuffle=False, pin_memory=True, batch_size=args.micro_bsz, num_workers=1, persistent_workers=False, drop_last=True)
    trainer.fit(model, data_loader)