import torch
import torch.nn as nn
import torchvision.transforms as transforms
from datetime import datetime
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import LinearLR
from datasets import load_dataset
from models.vit import ViT

def get_image_transforms(transform_fn):
    def img_transforms(example):
        example["pixel_values"] = transform_fn(example["image"])
        return example
    return img_transforms

def main():
    # Create/Download dataset
    tiny_imagenet_dataset = load_dataset("zh-plus/tiny-imagenet")

    # Format dataset
    img_to_tensor = transforms.ToTensor()
    tiny_imagenet_dataset_preproc = tiny_imagenet_dataset.map(get_image_transforms(img_to_tensor)) #batched for some reason performs worse ???

    # Create dataloaders
    train_dataloader = DataLoader(tiny_imagenet_dataset_preproc["train"], batch_size=64)
    val_dataloader = DataLoader(tiny_imagenet_dataset_preproc["valid"], batch_size=64)

    # Load model
    model = ViT(num_classes=500, in_channels=3, latent_vector_size=200, patch_size=8, num_heads=10)
    
    if torch.cuda.is_available():
        model.to("cuda")

    # Initialize optimizers and schedulers if desired
    optimizer = Adam(model.parameters(), weight_decay=0.1, lr=0.0008)
    scheduler = LinearLR(optimizer)

    # Define loss function
    loss_fn = nn.CrossEntropyLoss()

    # Train loop 
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    epoch_number = 0
    best_vloss = 1_000_000.

    for epoch in range(7):
        print('EPOCH {}:'.format(epoch_number + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = train_one_epoch(epoch_number)

        running_vloss = 0.0
        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        model.eval()

        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, vdata in enumerate(val_dataloader):
                vinputs, vlabels = vdata
                voutputs = model(vinputs)
                vloss = loss_fn(voutputs, vlabels)
                running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

    # Track best performance, and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = 'model_{}_{}'.format(timestamp, epoch_number)
        torch.save(model.state_dict(), model_path)

    epoch_number += 1

def train_one_epoch(train_dataloader, optimizer, model, loss_fn, epoch_index):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, batch in enumerate(train_dataloader):
        # Every data instance is an input + label pair
        labels, inputs = batch["label"], batch["pixel_values"]

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000 # running avg loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(train_dataloader) + i + 1
            print('Loss/train', last_loss, tb_x)
            running_loss = 0.

if __name__ == "__main__": 
    main()