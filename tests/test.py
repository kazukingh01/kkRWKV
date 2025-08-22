import torch
import lightning as L
from torch.utils.data import DataLoader
from lightning.pytorch import seed_everything
from lightning.pytorch import loggers as pl_loggers

from kkRWKV.dataset import RandomDataset
from kkRWKV.trainer import TrainCallback


if __name__ == "__main__":
    seed_everything(1)

    # data
    n_features = 10
    n_symbols  = 5
    n_label    = 5
    seq_len    = 128
    dataset_train = RandomDataset(n_features, n_symbols, n_label=n_label, seq_len=seq_len, n_samples=10000)
    dataset_valid = RandomDataset(n_features, n_symbols, n_label=n_label, seq_len=seq_len, n_samples=10000)
    data_loader_train = DataLoader(dataset_train, shuffle=False, pin_memory=True, batch_size=128, num_workers=4, persistent_workers=True, drop_last=True)
    data_loader_valid = DataLoader(dataset_valid, shuffle=False, pin_memory=True, batch_size=128, num_workers=4, persistent_workers=True, drop_last=True)

    # model for training
    from kkRWKV.model import RWKV, RWKV_FOR_TRAINING
    model = RWKV_FOR_TRAINING(
        n_features, n_symbols,
        seq_len=seq_len, num_classes=n_label,
        embd_dim=128, n_layers=3,
    )
    logger  = pl_loggers.TensorBoardLogger(save_dir="logs/")
    trainer = L.Trainer(
        max_epochs=1,
        accelerator="gpu", devices=1,
        precision=f"{model.mode_float.replace('fp', '')}-true",
        logger=logger,
        log_every_n_steps=10,
        callbacks=[TrainCallback(epoch_save=1)],
    )
    model.load_state_dict(model.generate_init_weight(), strict=True)
    trainer.fit(model, train_dataloaders=data_loader_train, val_dataloaders=data_loader_valid)

    # test
    testdata   = torch.stack([x[0] for x in dataset_valid])[:2]
    model_path = "./out/rwkv-epoch-0.pth"

    # load after training
    model.load_state_dict(torch.load(model_path)["state_dict"])
    print(model.eval().to("cuda")(testdata.to(model.dtype).to("cuda")))

    # inference by GPU
    model1 = RWKV.load_from_model_path("./out/rwkv-epoch-0.pth", num_classes=n_label, mode_float="fp32", is_cpu=False, is_jit=True)
    print(model1.eval().to("cuda")(testdata.to(torch.float32).to("cuda")))

    # inference by CPU
    from kkRWKV.model import RWKV_FOR_INFERENCE
    model2 = RWKV_FOR_INFERENCE("./out/rwkv-epoch-0.pth", dtype=torch.float32)
    print(model2.forward(testdata[0].to(torch.float32), seq_len=None))


