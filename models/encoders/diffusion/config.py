from typing import Optional, List
from pydantic import StrictStr, StrictInt, StrictFloat, StrictBool


import models.encoders.diffusion.diffusion
import models.encoders.diffusion.model


class Dataset(Config):
    name: StrictStr
    path: StrictStr
    resolution: StrictInt


class Diffusion(Config):
    beta_schedule: Instance


class Training(Config):
    n_iter: StrictInt
    optimizer: Optimizer
    scheduler: Optional[Scheduler]
    dataloader: DataLoader


class Eval(Config):
    wandb: StrictBool
    save_every: StrictInt
    valid_every: StrictInt
    log_every: StrictInt


class DiffusionConfig(MainConfig):
    dataset: Dataset
    model: Instance
    diffusion: Diffusion
    training: Training
    evaluate: Eval