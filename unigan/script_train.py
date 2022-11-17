""" The train script. """

import config
from dataloader import generate_data
from modellib.trainer import Trainer, TrainerConditional


if __name__ == '__main__':
    # 1. Generate config.
    cfg = config.ConfigTrainModel()
    # 2. Generate model.
    trainer = (Trainer if not hasattr(cfg.args, 'n_classes') else TrainerConditional)(cfg)
    # 3. Train
    trainer.train_model(train_data=generate_data(cfg))
