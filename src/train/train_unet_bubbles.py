from src.rplan_masks.unet_bubble_trainer import UnetBubbleTrainer


def train_unet_bubbles():
    trainer = UnetBubbleTrainer(lr=1e-4, mask_size=64, epochs=100, batch_size=16)
    trainer.train()

if __name__ == '__main__':
    train_unet_bubbles()
