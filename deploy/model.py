from torch import nn
import torch
from torchvision import transforms
from PIL import Image


model = nn.Sequential(
        # Conv2d( in_channels, out_channels, kernel_size) B,3,32,32
        nn.Conv2d(3, 64, 3, stride=1, padding=1), nn.BatchNorm2d(64), nn.LeakyReLU(0.1), nn.MaxPool2d(2, 2),
        # B,32,16,16
        nn.Conv2d(64, 128, 3, stride=1, padding=1), nn.BatchNorm2d(128), nn.LeakyReLU(0.1), nn.MaxPool2d(2, 2),
        # B,32,8,8
        nn.Conv2d(128, 256, 3, stride=1, padding=1), nn.BatchNorm2d(256), nn.LeakyReLU(0.1), nn.MaxPool2d(2, 2),
        # B,32,4,4
        nn.Conv2d(256, 512, 3, stride=1, padding=1), nn.BatchNorm2d(512), nn.LeakyReLU(0.1),
        nn.AdaptiveAvgPool2d(1),  ##like in ResNET
        nn.Flatten(),
        nn.Linear(512, 512), nn.BatchNorm1d(512), nn.LeakyReLU(0.1),
        nn.Linear(512, 10)
    )


def inference(model, image_path, device='cpu'):
    """
    Predict the class of an image using the given model.

    Args:
        model (torch.nn.Module): The trained model.
        image_path (str): The path to the image file.
        class_names (list): A list of class names corresponding to the model's output.
        device (str): The device to run the model on ('cuda' or 'cpu').

    Returns:
        str: The predicted class name.
    """
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    # Load the image
    image = Image.open(image_path).convert('RGB')

    # Define the preprocessing transforms
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    # Preprocess the image
    image = preprocess(image).unsqueeze(0).to(device)

    # Set the model to evaluation mode
    model.eval()

    # Make the prediction
    with torch.no_grad():
        logits = model(image)
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        predicted_class_idx = torch.argmax(probabilities, dim=1).item()

    # Get the predicted class name
    predicted_class_name = class_names[predicted_class_idx]

    return predicted_class_name


if __name__ == "__main__":
    model.load_state_dict(torch.load('best_model_checkpoint.pth'))
    model.to('cuda')

    image_path = 'test_img.png'
    predicted_class = inference(model, image_path)
    print(f"Predicted class: {predicted_class}")
