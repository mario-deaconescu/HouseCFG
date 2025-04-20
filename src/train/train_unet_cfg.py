from src.rplan_masks.unet_trainer import UnetTrainer


def train_unet_cfg():
    trainer = UnetTrainer(lr=1e-4, mask_size=64, epochs=100, batch_size=16)
    trainer.train()

if __name__ == '__main__':
    train_unet_cfg()
