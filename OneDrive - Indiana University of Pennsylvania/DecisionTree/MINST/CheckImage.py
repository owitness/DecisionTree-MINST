import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# Set random seed for reproducibility
torch.manual_seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Hyperparameters
batch_size = 64
learning_rate = 0.001
num_epochs = 5

# MNIST Dataset loading
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
])

# Load training and test datasets
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)

# Data loaders
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


# Define the neural network model
class LeNet(nn.Module):
    def __init__(self):
        # convolutional Layers
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)

        # Fully Connceted layers
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2)

        x = x.view(-1, 16 *4 * 4)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        x = self.fc3(x)
        return x 
    

# Initialize the model
model = LeNet().to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# Training the model
def train():
    model.train()
    total_step = len(train_loader)

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for i, (images, labels) in enumerate(train_loader):
            # Move tensors to the configured device
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{total_step}], Loss: {loss.item():.4f}')

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        print(f'Epoch {epoch + 1}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')


# Testing the model
def test():
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f'Test Accuracy: {100 * correct / total:.2f}%')
    return 100 * correct / total


# Function to visualize sample predictions with probabilities
def visualize_predictions():
    model.eval()

    # Get a batch of test images
    dataiter = iter(test_loader)
    images, labels = next(dataiter)

    # Move to device and get predictions
    images = images.to(device)
    outputs = model(images)
    probabilities = F.softmax(outputs, dim=1).cpu().detach().numpy()

    # Get predicted classes
    _, predicted = torch.max(outputs, 1)
    predicted = predicted.cpu().numpy()

    # Plot 10 sample images
    fig = plt.figure(figsize=(15, 8))
    for i in range(10):
        # Plot the image
        ax1 = fig.add_subplot(2, 10, i + 1)
        img = images[i].cpu().squeeze().numpy()
        ax1.imshow(img, cmap='gray')
        ax1.set_title(f"True: {labels[i]}\nPred: {predicted[i]}")
        ax1.axis('off')

        # Plot the probabilities
        ax2 = fig.add_subplot(2, 10, i + 11)
        ax2.bar(np.arange(10), probabilities[i])
        ax2.set_xticks(np.arange(10))
        ax2.set_ylim(0, 1)
        if i == 0:
            ax2.set_ylabel('Probability')
        if i == 4:
            ax2.set_xlabel('Digit Class')

    plt.tight_layout()
    plt.savefig('mnist_predictions.png')
    plt.show()


# Function to load and predict on a custom image
def predict_custom_image(image_path, model_path=None):
    """
    Load an image, preprocess it to MNIST format, and visualize prediction probabilities

    Args:
        image_path (str): Path to the image file
        model_path (str, optional): Path to saved model weights. If None, uses current model state
    """
    from PIL import Image
    import os

    # Load model if path is provided
    if model_path and os.path.exists(model_path):
        model= torch.load(model_path, map_location=device, weights_only = False)
        print(f"Loaded model from {model_path}")

    model.eval()

    # Load and preprocess the image
    print(f"Loading image from {image_path}")
    try:
        # Open image and convert to grayscale
        original_image = Image.open(image_path).convert('L')

        # Display original image
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(original_image, cmap='gray')
        plt.title("Original Image")
        plt.axis('off')

        # Resize to 28x28 (MNIST size)
        resized_image = original_image.resize((28, 28), Image.Resampling.LANCZOS)

        # Display resized image
        plt.subplot(1, 3, 2)
        plt.imshow(resized_image, cmap='gray')
        plt.title("Resized to 28x28")
        plt.axis('off')

        # Convert to numpy array and normalize
        img_array = np.array(resized_image)

        # Invert if needed (MNIST has white digits on black background)
        # Check if image appears to be inverted compared to MNIST format
        if img_array.mean() > 128:  # If image is predominantly white
            img_array = 255 - img_array  # Invert colors

        # Apply same normalization as during training
        img_normalized = (img_array / 255.0 - 0.1307) / 0.3081

        # Display normalized image
        plt.subplot(1, 3, 3)
        plt.imshow(img_normalized, cmap='gray')
        plt.title("Normalized")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

        # Convert to tensor and add batch and channel dimensions
        img_tensor = torch.tensor(img_normalized, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        # Move to device and predict
        img_tensor = img_tensor.to(device)
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = F.softmax(outputs, dim=1).cpu().numpy()[0]
            prediction = torch.argmax(outputs, dim=1).item()

        # Plot probabilities
        plt.figure(figsize=(10, 6))

        # Show the processed image
        plt.subplot(1, 2, 1)
        plt.imshow(img_normalized, cmap='gray')
        plt.title(f"Prediction: {prediction}")
        plt.axis('off')

        # Plot probability distribution
        plt.subplot(1, 2, 2)
        bars = plt.bar(np.arange(10), probabilities)
        plt.xticks(np.arange(10))
        plt.xlabel('Digit Class')
        plt.ylabel('Probability')
        plt.title('Class Probabilities')
        plt.ylim(0, 1)

        # Highlight the predicted class
        bars[prediction].set_color('red')

        plt.tight_layout()
        plt.show()

        return prediction, probabilities

    except Exception as e:
        print(f"Error processing image: {e}")
        return None, None


# Main function
def main():
    print("Please give an image path to look at")
    image_path = input()
    model_path = 'model.pt'

    predict_custom_image(image_path, model_path=model_path)
    # Save the model


if __name__ == "__main__":
    main()