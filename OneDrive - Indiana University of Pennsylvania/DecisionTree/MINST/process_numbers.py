import os
import glob
from prepare_image import prepare_for_mnist

def process_all_numbers(input_dir, output_dir):
    """
    Process all number images in the input directory and its subdirectories and save them in MNIST format
    
    Parameters:
    - input_dir: Directory containing subdirectories of cropped number images
    - output_dir: Directory where processed images will be saved (preserving subdirectory structure)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all subdirectories (each representing a digit)
    subdirs = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    total_images = 0
    
    print(f"Found {len(subdirs)} digit directories")
    
    # Process each subdirectory
    for subdir in subdirs:
        input_subdir = os.path.join(input_dir, subdir)
        output_subdir = os.path.join(output_dir, subdir)
        
        # Create corresponding output subdirectory
        os.makedirs(output_subdir, exist_ok=True)
        
        # Get all image files in the current subdirectory
        image_files = glob.glob(os.path.join(input_subdir, "*.png")) + \
                     glob.glob(os.path.join(input_subdir, "*.jpg")) + \
                     glob.glob(os.path.join(input_subdir, "*.jpeg"))
        
        print(f"\nProcessing digit {subdir}: Found {len(image_files)} images")
        total_images += len(image_files)
        
        # Process each image in the current subdirectory
        for img_path in image_files:
            # Get the filename without extension
            filename = os.path.splitext(os.path.basename(img_path))[0]
            output_path = os.path.join(output_subdir, f"{filename}_mnist.png")
            
            try:
                # Process the image
                processed_image = prepare_for_mnist(
                    image_path=img_path,
                    output_path=output_path,
                    display=False  # Set to True if you want to see each image being processed
                )
                print(f"Successfully processed {subdir}/{filename}")
            except Exception as e:
                print(f"Error processing {subdir}/{filename}: {str(e)}")
    
    print(f"\nCompleted processing {total_images} images across {len(subdirs)} digit directories")

if __name__ == "__main__":
    # Define input and output directories
    input_directory = "DecisionTree/MINST/mnist_images/train"  # Directory containing digit subdirectories
    output_directory = "DecisionTree/MINST/mnist_images/processed"
    
    # Process all images
    process_all_numbers(input_directory, output_directory) 