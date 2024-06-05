import torch
from tqdm import tqdm
import config
import matplotlib.pyplot as plt
import numpy as np


def intersection_over_union(pred, true):
    """
    Calculates intersection and union for a batch of images.

    Args:
        pred (torch.Tensor): a tensor of predictions
        true (torc.Tensor): a tensor of labels

    Returns:
        intersection (int): total intersection of pixels
        union (int): total union of pixels
    """
    valid_pixel_mask = true.ne(255)  # valid pixel mask
    true = true.masked_select(valid_pixel_mask).to("cpu")
    pred = pred.masked_select(valid_pixel_mask).to("cpu")

    # Intersection and union totals
    intersection = np.logical_and(true, pred)
    union = np.logical_or(true, pred)
    return intersection.sum() / union.sum()


def train_fn(train_loader, model, optimizer, loss_fn):
    model.train() 
    train_loader = tqdm(train_loader, desc="batches")
    for it in train_loader:
        data = it["chip"].float().to(config.device)
        targets = it["label"].float().to(config.device)
        optimizer.zero_grad()
        predictions = model(data)
        loss = loss_fn(predictions, targets)
        
        loss.backward()
        optimizer.step()

        train_loader.set_postfix(loss=loss.item())


def val_fn(val_loader, model):
    iou_list = []
    val_loader = tqdm(val_loader)
    with torch.no_grad():
        model.eval()
        for it in val_loader:
            input_image = it["chip"].type(torch.FloatTensor).to(config.device)
            true_mask = it["label"].squeeze()
            predicted_mask = model(input_image)
            batch_iou = intersection_over_union(predicted_mask.detach().to("cpu"), true_mask)
            iou_list.append(batch_iou)
            val_loader.set_postfix(iou=sum(iou_list) / len(iou_list))
    model.train()
    return sum(iou_list) / len(iou_list)
