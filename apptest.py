import torch
import utils
import config
import matplotlib.pyplot as plt
from data import CloudDataset
import pandas as pd
from torch.utils.data import DataLoader
import config
import utils

model = config.model
model.to(config.device)
optimizer = config.optimizer(model.parameters(), lr=config.learning_rate)

model.load_state_dict(torch.load("model.pth"))
optimizer.load_state_dict(torch.load("optim.pth"))

model.eval()

val_x = pd.read_csv("data/val_x.csv").drop(columns="Unnamed: 0")
val_y = pd.read_csv("data/val_y.csv").drop(columns="Unnamed: 0")
validation_dataset = CloudDataset(val_x, config.bands, val_y, config.val_transforms)
val_loader = DataLoader(
    dataset=validation_dataset,
    batch_size=config.val_batch_size,
    shuffle=False,
    pin_memory=True,
    num_workers=0,
)


def app(val_loader, model, sample):
    with torch.no_grad():
        for it in val_loader:
            input_image = it["chip"].type(torch.FloatTensor).to(config.device)
            true_mask = it["label"].squeeze()

            print(input_image[sample].max())
            print(input_image[sample].mean())
            print(input_image[sample].median())

            predicted_mask = model(input_image)

            utils.plot_masks(
                torch.round(predicted_mask[sample]).detach().cpu().numpy(),
                true_mask[sample].detach().cpu().numpy(),
            )

            break


app(val_loader, model)
