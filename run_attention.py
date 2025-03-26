from pytorch_lightning import callbacks
from datamodule.ChexpertDataModule import CheXpertDataModule
import pytorch_lightning as pl
import pandas as pd
from pathlib import Path
from pytorch_lightning.callbacks import ModelCheckpoint
from model.Vit_pretrained import pretrained_vit
from model.deit import DeiT


def train():
    pl.seed_everything(42)
    pathFileTrain = Path(__file__).parents[2].joinpath("./CheXpert-v1.0-small/chex_train.csv")
    pathFileValid = Path(__file__).parents[2].joinpath("./CheXpert-v1.0-small/chex_val.csv")
    pathFileTest = Path(__file__).parents[2].joinpath("./CheXpert-v1.0-small/chex_test.csv")
    CheXpertData = CheXpertDataModule(pathFileTrain = pathFileTrain, pathFileValid = pathFileValid, pathFileTest = pathFileTest)
    #model = pretrained_vit(out_size=5)
    model = DeiT(out_size=5)
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode='min', filename = "Attention-{epoch:02d}-{val_loss:.2f}", save_last=True)
    trainer = pl.Trainer(max_epochs=5, gpus=1, fast_dev_run = False, check_val_every_n_epoch = 1, callbacks = [checkpoint_callback])
    trainer.fit(model, CheXpertData)
    trainer.test(model, CheXpertData)



if __name__ == '__main__':
    train()