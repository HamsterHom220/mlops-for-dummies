from torch import nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
import torch
import torch.nn.functional as F
from tqdm import tqdm

import sys
import os
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT_DIR)
from deploy.model import model as deploy_model

def train(model=deploy_model):
    # Define device (GPU/CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=0.0003, weight_decay=0.0005)
    scheduler = StepLR(optim, step_size=10, gamma=0.5)

    # Define transformations
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.1, 0.1, 0.1),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        transforms.RandomErasing(p=0.5,scale=(0.02, 0.1),value=1.0, inplace=False),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    batch_size=64

    ## Train with the good old torchvision set

    cifar10 = torchvision.datasets.CIFAR10(root=os.path.join(ROOT_DIR, 'data'), download=False, train=True, transform=transform_train)
    cifar10_test = torchvision.datasets.CIFAR10(root=os.path.join(ROOT_DIR, 'data'), download=False, train=False, transform=transform_test)

    n = batch_size
    dataloader = DataLoader(cifar10, batch_size=batch_size, shuffle=True, num_workers=2)
    dataloader_test = DataLoader(cifar10_test, batch_size=batch_size, shuffle=False, num_workers=2)

    ## xavier init
    def init_weights_xavier(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)  # Apply Xavier normal initialization
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    model.apply(init_weights_xavier)

    i = 0
    losses = []
    steps = []
    losses_t = []

    # Initialize variables to track the best loss and patience
    best_loss = float('inf')
    patience = 5  # Number of epochs to wait for improvement before stopping
    patience_counter = 0

    for epoch in range(32):
        epoch_loss = 0
        for batch in tqdm(dataloader):
            i += 1
            x = batch[0].to('cuda')
            y = batch[1].to('cuda')
            model = model.to('cuda')

            logits = model(x)
            loss = F.cross_entropy(logits, y)
            optim.zero_grad()

            loss.backward()
            optim.step()

            epoch_loss += loss.item()
            losses.append(loss.item())
            losses_t.append(loss.detach())
            steps.append(i)

            if i % 1000 == 0:
                print(loss)

        # Calculate the average loss for the epoch
        epoch_loss /= len(dataloader)

        # Check if the current epoch loss is better than the best loss
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience_counter = 0
            # Save the model checkpoint
            torch.save(model.state_dict(), 'best_model_checkpoint.pth')
            print(f"Epoch {epoch + 1}: Improved loss to {best_loss}. Saving model checkpoint.")
        else:
            patience_counter += 1
            print(f"Epoch {epoch + 1}: Loss did not improve. Patience counter at {patience_counter}.")

        # Check if patience has been exceeded
        if patience_counter >= patience:
            print(f"Epoch {epoch + 1}: Loss has not improved for {patience} epochs. Stopping training.")
            break

        scheduler.step()


    total_correct = 0
    total_predictions = 0

    # Loop over training dataset
    for x_batch, y_batch in dataloader:
        logits = model(x_batch.to("cuda")) # Forward pass on the mini-batch
        loss = F.cross_entropy(logits.cpu(), y_batch) # Compute loss

        # Calculate predictions for the batch
        pred_labels = torch.max(logits, dim=1).indices

        # Update total correct predictions and total predictions
        total_correct += (y_batch == pred_labels.cpu()).sum().item()
        total_predictions += y_batch.size(0)

    # Calculate overall accuracy
    overall_accuracy = total_correct / total_predictions
    print(f"Training Accuracy: {overall_accuracy}")

    total_correct = 0
    total_predictions = 0

    # Loop over testing dataset
    for x_batch, y_batch in dataloader_test:
        logits = model(x_batch.to(device)) # Forward pass on the mini-batch
        loss = F.cross_entropy(logits.cpu(), y_batch) # Compute loss

        # Calculate predictions for the batch
        pred_labels = torch.max(logits, dim=1).indices

        # Update total correct predictions and total predictions
        total_correct += (y_batch == pred_labels.cpu()).sum().item()
        total_predictions += y_batch.size(0)

    # Calculate overall accuracy and print it out.
    overall_accuracy = total_correct / total_predictions
    #DON'T FORGET TO PRINT OUT YOUR TESTING ACCURACY
    print(f"Testing Accuracy: {overall_accuracy}")


if __name__ == "__main__":
    train()