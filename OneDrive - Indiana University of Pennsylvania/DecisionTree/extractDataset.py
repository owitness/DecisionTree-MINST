import os
import torch
from torchvision import datasets, transforms
from PIL import Image

def extract_mnist_dataset():
    # Create a directory to save the images
    base_dir = "mnist_images"
    
    # Create directories for train and test sets
    train_dir = os.path.join(base_dir, "train")
    test_dir = os.path.join(base_dir, "test")
    
    # Download and load MNIST dataset
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
    
    # Create class directories
    for split_dir in [train_dir, test_dir]:
        if not os.path.exists(split_dir):
            os.makedirs(split_dir)
        
        # Create directories for each class (0-9)
        for i in range(10):
            class_dir = os.path.join(split_dir, str(i))
            if not os.path.exists(class_dir):
                os.makedirs(class_dir)
    
    # Extract and save training images
    print("Extracting training images...")
    for idx, (image, label) in enumerate(train_dataset):
        # Convert tensor to PIL Image
        image_pil = transforms.ToPILImage()(image)
        
        # Save the image
        image_path = os.path.join(train_dir, str(label), f"{idx}.png")
        image_pil.save(image_path)
        
        if idx % 1000 == 0:
            print(f"Processed {idx}/{len(train_dataset)} training images")
    
    # Extract and save test images
    print("Extracting test images...")
    for idx, (image, label) in enumerate(test_dataset):
        # Convert tensor to PIL Image
        image_pil = transforms.ToPILImage()(image)
        
        # Save the image
        image_path = os.path.join(test_dir, str(label), f"{idx}.png")
        image_pil.save(image_path)
        
        if idx % 1000 == 0:
            print(f"Processed {idx}/{len(test_dataset)} test images")
    
    print("Extraction complete!")
    print(f"Training images: {len(train_dataset)}")
    print(f"Test images: {len(test_dataset)}")
    print(f"Images saved to {os.path.abspath(base_dir)}")

if __name__ == "__main__":
    extract_mnist_dataset()