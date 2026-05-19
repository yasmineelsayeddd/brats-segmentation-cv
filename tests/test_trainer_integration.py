import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models import UNet
from src.training.losses import DiceCELoss
from src.training.trainer import Trainer


def test_one_epoch_training_smoke(tmp_path):
    images = torch.randn(4, 4, 32, 32)
    masks = torch.randint(0, 4, (4, 32, 32))
    loader = DataLoader(TensorDataset(images, masks), batch_size=2)
    model = UNet(base_channels=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer = Trainer(
        model=model,
        criterion=DiceCELoss(),
        optimizer=optimizer,
        device="cpu",
        checkpoint_dir=tmp_path / "checkpoints",
        output_dir=tmp_path / "outputs",
        experiment_name="smoke",
        early_stopping_patience=2,
    )
    history = trainer.fit(loader, loader, epochs=1)
    assert len(history) == 1
    assert (tmp_path / "checkpoints" / "smoke_best.pth").exists()
    assert (tmp_path / "outputs" / "smoke" / "history.csv").exists()
