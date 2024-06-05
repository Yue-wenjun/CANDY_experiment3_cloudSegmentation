import torch
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from candy import CANDY, Multi_CANDY, Cross_CANDY

epochs = 12
train_batch_size = 32
val_batch_size = 32
device = "cuda:0" if torch.cuda.is_available() else "cpu"
bands = ["B02", "B03", "B04", "B08"]
loss_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam
model = Multi_CANDY(
    in_channel=4, batch_size=32, out_channel=1, input_size=512, hidden_size=512
).to(device)

scaler = torch.cuda.amp.GradScaler()
learning_rate = 3e-4
train_transforms = A.Compose(
    [
        A.Rotate(limit=60, p=0.6),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        ToTensorV2(),
    ],
)
val_transforms = A.Compose(
    [
        ToTensorV2(),
    ]
)
