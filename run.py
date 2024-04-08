import copy
import pytorch_lightning as pl
from meter.config import ex
from meter.modules import METERTransformerSS
from data import F30kDataModule, MscocoDataModule
import torch
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'


@ex.automain
def main(_config):

    _config = copy.deepcopy(_config)
    pl.seed_everything(_config["seed"], workers=True)
    print(_config)

    if 'f30k' in _config['exp_name']:
        dm = F30kDataModule(_config)
    else:
        dm = MscocoDataModule(_config)

    model = METERTransformerSS(_config)

    if _config['test_only']:
        ckpt = torch.load(_config['checkpoint'], map_location="cuda:0")
        model.load_state_dict(ckpt['state_dict'])

    exp_name = f'{_config["exp_name"]}'

    os.makedirs(_config["log_dir"], exist_ok=True)
    
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=5,
        dirpath = _config['save_path'],
        monitor="best_irtr",
        mode="max",
        save_last=True,
    )
    logger = pl.loggers.TensorBoardLogger(
        _config["log_dir"],
        name=f'{exp_name}_seed{_config["seed"]}_from_{_config["load_path"].split("/")[-1][:-5]}',
        version=_config['experiment_name']
    )

    callbacks = [checkpoint_callback]

    num_gpus = (
        _config["num_gpus"]
        if isinstance(_config["num_gpus"], int)
        else len(_config["num_gpus"])
    )

    trainer = pl.Trainer(
        gpus=[0],
        precision=_config["precision"],
        #accelerator="ddp",s
        # accelerator='ddp',
        # strategy='ddp',
        benchmark=True,
        deterministic=True,
        max_epochs=_config["max_epoch"],
        callbacks=callbacks,
        logger=logger,
        #replace_sampler_ddp=False,
        log_every_n_steps=10,
        flush_logs_every_n_steps=10,
        weights_summary="top",
        val_check_interval=_config["val_check_interval"],
        # gradient_clip_val=2.0
    )
    # print("***********************{}".format(trainer.global_rank)s)
    # if trainer.global_rank == 0:s
    #     print(_config)

    if not _config["test_only"]:
        trainer.fit(model, datamodule=dm)
    else:
        trainer.test(model, datamodule=dm)

