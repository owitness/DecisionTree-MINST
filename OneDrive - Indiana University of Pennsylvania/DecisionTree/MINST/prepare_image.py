import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def prepare_for_mnist(image_path, output_path=None, display=True):
    """
    Process an image to match MNIST format (28x28 grayscale with white digit on black background)
    
    Parameters:
    - image_path: Path to the input image or numpy array from camera
    - output_path: If provided, saves the processed image to this path
    - display: Whether to display the original and processed images
    
    Returns:
    - processed_image: 28x28 numpy array ready for MNIST model input
    """
    # Read the image if path is provided, otherwise assume it's already a numpy array
    if isinstance(image_path, str):
        img = cv2.imread(image_path)
    else:
        img = image_path
    
    # Convert to grayscale if it's not already
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    # Apply thresholding to get black background and white digit
    # The MNIST dataset has white digits on a black background
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find contours to identify the digit
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # If contours are found, crop to the bounding box of the largest contour
    if contours:
        # Find the largest contour (presumably the digit)
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Add a small padding
        padding = 5
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(binary.shape[1] - x, w + 2 * padding)
        h = min(binary.shape[0] - y, h + 2 * padding)
        
        # Crop to the digit
        cropped = binary[y:y+h, x:x+w]
    else:
        cropped = binary
    
    # Resize to 20x20 (standard for MNIST content, which is centered in 28x28)
    resized = cv2.resize(cropped, (20, 20), interpolation=cv2.INTER_AREA)
    
    # Create a 28x28 blank (black) image
    mnist_img = np.zeros((28, 28), dtype=np.uint8)
    
    # Calculate center position to paste the resized image
    offset_x = (28 - 20) // 2
    offset_y = (28 - 20) // 2
    
    # Place the resized image in the center of the 28x28 image
    mnist_img[offset_y:offset_y+20, offset_x:offset_x+20] = resized
    
    # Normalize to match MNIST format (0-1 float values)
    mnist_normalized = mnist_img / 255.0
    
    # Display the original and processed images if requested
    if display:
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 3, 1)
        plt.title("Original")
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if len(img.shape) == 3 else img, cmap='gray')
        
        plt.subplot(1, 3, 2)
        plt.title("Processed")
        plt.imshow(mnist_img, cmap='gray')
        
        plt.subplot(1, 3, 3)
        plt.title("MNIST Format (Normalized)")
        plt.imshow(mnist_normalized, cmap='gray')
        plt.tight_layout()
        plt.show()
    
    # Save the processed image if requested
    if output_path:
        Image.fromarray(mnist_img).save(output_path)
    
    return mnist_normalized

prepare_for_mnist("path_to_your_cropped_image.png", "example.png")