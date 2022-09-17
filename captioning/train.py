import math
import os
import sys
import warnings

import numpy as np
import torch
import wandb
from data.transformer_dataloader import get_data_loader
from prediction import predict
from tqdm import tqdm
from utils.models import caption

from captioning.utils.utils import NestedTensor
from configuration import transformers_config, init_wab

warnings.filterwarnings("ignore")


def train_one_epoch(model, criterion, data_loader, optimizer, device, max_norm):
    """
    One epoch's training.
    :param model: (torch.nn.Module) the neural network
    :param criterion: (torch.nn) loss function
    :param data_loader: (DataLoader) a torch.utils.data.DataLoader object that fetches data
    :param optimizer: (torch.optim) optimizer which updates the weights of model
    :param device: (torch.device) device - 'cuda' or 'cpu'
    :param max_norm: (float) value of max norm for gradient clipping
    :return: (float) mean training loss of this epoch
    """
    model.train()
    criterion.train()

    epoch_loss = 0.0
    total = len(data_loader)

    with tqdm(total=total) as pbar:
        for images, masks, caps, cap_masks in data_loader:
            samples = NestedTensor(images, masks).to(device)
            caps = caps.to(device)
            cap_masks = cap_masks.to(device)

            outputs = model(samples, caps[:, :-1], cap_masks[:, :-1])
            loss = criterion(outputs.permute(0, 2, 1), caps[:, 1:])
            loss_value = loss.item()
            epoch_loss += loss_value

            if not math.isfinite(loss_value):
                print(f"Loss is {loss_value}, stopping training")
                sys.exit(1)

            optimizer.zero_grad()
            loss.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

            pbar.update(1)

    return epoch_loss / total


@torch.no_grad()
def evaluate(model, criterion, data_loader, device):
    model.eval()
    criterion.eval()

    validation_loss = 0.0
    total = len(data_loader)

    with tqdm(total=total, leave=False) as pbar:
        for imgs, masks, captions, cap_masks in data_loader:
            samples = NestedTensor(imgs, masks).to(device)
            captions = captions.to(device)
            cap_masks = cap_masks.to(device)

            outputs = model(samples, captions[:, :-1], cap_masks[:, :-1])
            loss = criterion(outputs.permute(0, 2, 1), captions[:, 1:])

            validation_loss += loss.item()
            pbar.update(1)

    return validation_loss / total


def train(config):
    device = torch.device(config.device)
    print(f"Initializing Device: {device}")

    # load dataloader
    training_dataloader, validation_dataloader, test_dataloader = get_data_loader(
        image_path=config.args.data_path + "/images",
        report_address=config.args.data_path + "/Pneumothorax_reports.csv",
        instance_uid_adress=config.args.data_path + "/Pneumothorax_reports.csv",
        batch_size=config.batch_size,
    )

    # config.vocab_size = len(training_dataloader.dataset.dataset.vocab)
    # config.vocab = training_dataloader.dataset.dataset.vocab

    # model, criterion = Caption.build_model(config,training_dataloader.dataset.dataset.vocab.stoi["<PAD>"])
    model, criterion = caption.build_model(config)
    model.to(device)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of params: {n_parameters}")

    param_dicts = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if "backbone" not in n and p.requires_grad
            ]
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if "backbone" in n and p.requires_grad
            ],
            "lr": config.lr_backbone,
        },
    ]

    optimizer = torch.optim.AdamW(
        param_dicts, lr=config.lr, weight_decay=config.weight_decay
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, config.lr_drop)

    # if os.path.exists(config.checkpoint):
    #     print("Loading Checkpoint...")
    #     checkpoint = torch.load(config.checkpoint, map_location="cuda")
    #     model.load_state_dict(checkpoint["model"])
    #     optimizer.load_state_dict(checkpoint["optimizer"])
    #     lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
    #     config.start_epoch = checkpoint["epoch"] + 1

    print("Start Training..")
    min_valid_loss = np.inf
    epochs_no_improve = 0
    early_stop = False

    for epoch in range(config.start_epoch, config.epochs):
        print(f"Epoch: {epoch}")
        epoch_loss = train_one_epoch(
            model,
            criterion,
            training_dataloader,
            optimizer,
            device,
            config.clip_max_norm,
        )
        lr_scheduler.step()
        print(f"Training Loss: {epoch_loss}")
        # torch.cuda.empty_cache()

        validation_loss = evaluate(model, criterion, validation_dataloader, device)

        # log for wandb if exists
        if config.args.enable_wab:
            wandb.log(
                {"epoch": epoch, "train_loss": epoch_loss, "val_loss": validation_loss}
            )
        print(f"Validation Loss: {validation_loss}")

        if min_valid_loss > validation_loss:
            print(
                f"Validation Loss Decreased({min_valid_loss:.6f}--->{validation_loss:.6f}) \t Saving The Model"
            )
            epochs_no_improve = 0
            min_valid_loss = validation_loss
            if config.save_model:
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "epoch": epoch,
                    },
                    config.checkpoint,
                )
                torch.save(
                    model.backbone._modules["0"].state_dict(),
                    config.backbone_checkpoint,
                )

        else:
            epochs_no_improve += 1
            print(f"Validation Loss Not Decreased...")
        if epoch > 5 and epochs_no_improve == config.early_stopping_patience:
            print("Early stopping!")
            early_stop = True
            break
        else:
            continue

    test_result = predict(model, test_dataloader, device)
    print(test_result)

    # Call wandb.finish() to upload your TensorBoard logs to W&B
    wandb.finish()


if __name__ == "__main__":
    # load config
    config = transformers_config()

    # initialize wandb
    if config.args.wab_project is None:
        val_loss_filter = "_".join(config.args.validation_loss)
        config.args.wab_project = f"Image_Captioning_{config.args.dataset_name}_Dataset.{config.args.model}.{config.args.image_size}_Img_size.{val_loss_filter}_loss"
    if config.args.enable_wab:
        init_wab(
            wab_config_path=config.args.wab_config_path,
            model_config=config.backbone_cfg,
            model_args=config.args,
            entity=config.args.wab_entity,
            project_name=config.args.wab_project,
            key=config.args.wab_key,
        )

    # train based on the config
    train(config)
