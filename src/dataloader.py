from torch.utils.data import DataLoader, random_split
import torch


def get_dataloaders(
    dataset,
    batch_size=32,
    val_split=0.2,
    num_workers=0,
    seed=42
):
    """
    Split dataset into training and validation sets
    and create DataLoaders.
    """

    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size

    generator = torch.Generator().manual_seed(seed)

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=generator
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader
