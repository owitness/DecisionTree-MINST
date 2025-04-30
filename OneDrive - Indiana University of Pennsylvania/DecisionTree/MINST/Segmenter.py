import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from scipy.signal import find_peaks


def image_histogram_analysis(img_array, visualize = True):
    """
    Takes an image file path, calculates the sum of each row and column,
    and creates histograms for both.
    
    Args:
        img_array (str): numpy array representing the grayscale image
        
    Returns:
        None: Displays the histograms
    """
    num_peaks = 10
    
    # Get the dimensions of the image
    height, width = img_array.shape

    min_distance_row = width/20
    min_distance_col = height/20
    
    # Calculate the sum of each row and column
    row_sums = np.sum(img_array, axis=1)
    col_sums = np.sum(img_array, axis=0)
    
    # Find peaks in row and column sums with minimum distance constraint
    row_peaks, row_properties = find_peaks(row_sums, height=0, distance=min_distance_row)
    col_peaks, col_properties = find_peaks(col_sums, height=0, distance=min_distance_col)
    
    # If the above approach doesn't find enough peaks, we'll implement a greedy approach
    # to select well-spaced peaks manually
    
    # Function to select well-spaced peaks from all detected peaks
    def get_well_spaced_peaks(values, all_peaks, n_peaks, min_spacing):
        if len(all_peaks) <= n_peaks:
            return all_peaks
        
        # Get peak heights
        peak_heights = values[all_peaks]
        
        # Sort peaks by height (highest first)
        sorted_idx = np.argsort(-peak_heights)
        sorted_peaks = all_peaks[sorted_idx]
        
        # Greedily select peaks
        selected_peaks = []
        for peak in sorted_peaks:
            # Check if this peak is far enough from already selected peaks
            if not any(abs(peak - sp) < min_spacing for sp in selected_peaks):
                selected_peaks.append(peak)
                if len(selected_peaks) >= n_peaks:
                    break
        
        # If we didn't get enough peaks, relax the distance constraint
        if len(selected_peaks) < n_peaks and min_spacing > 1:
            # Recursively try with a smaller spacing
            return get_well_spaced_peaks(values, all_peaks, n_peaks, min_spacing // 2)
            
        return np.array(selected_peaks)
    
    # Find all potential peaks first with minimal constraints
    all_row_peaks, _ = find_peaks(row_sums, height=0)
    all_col_peaks, _ = find_peaks(col_sums, height=0)
    
    # Select well-spaced peaks
    top_row_peaks = get_well_spaced_peaks(row_sums, all_row_peaks, num_peaks, min_distance_row)
    top_col_peaks = get_well_spaced_peaks(col_sums, all_col_peaks, num_peaks, min_distance_col)
    
    # Sort the final selection by position for easier interpretation
    top_row_peaks = np.sort(top_row_peaks)
    top_col_peaks = np.sort(top_col_peaks)
    
    if visualize:
        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Plot the row sums histogram
        ax1.bar(range(height), row_sums, color='blue', alpha=0.5)
        ax1.set_title(f'Row Sums Histogram with Top {len(top_row_peaks)} Well-Spaced Peaks')
        ax1.set_xlabel('Row Index')
        ax1.set_ylabel('Sum of Pixel Values')
        
        # Highlight the top peaks for rows
        ax1.plot(top_row_peaks, row_sums[top_row_peaks], 'ro', markersize=8)
        for i, peak in enumerate(top_row_peaks):
            ax1.annotate(f'{i+1}', (peak, row_sums[peak]), 
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=12, fontweight='bold')
        
        # Plot the column sums histogram
        ax2.bar(range(width), col_sums, color='green', alpha=0.5)
        ax2.set_title(f'Column Sums Histogram with Top {len(top_col_peaks)} Well-Spaced Peaks')
        ax2.set_xlabel('Column Index')
        ax2.set_ylabel('Sum of Pixel Values')
        
        # Highlight the top peaks for columns
        ax2.plot(top_col_peaks, col_sums[top_col_peaks], 'ro', markersize=8)
        for i, peak in enumerate(top_col_peaks):
            ax2.annotate(f'{i+1}', (peak, col_sums[peak]), 
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=12, fontweight='bold')
        
        # Adjust layout
        plt.tight_layout()
        
        # Show plot
        plt.show()
        
        # Print the locations of the top peaks
        print(f"Top {len(top_row_peaks)} row peaks at indices: {top_row_peaks}")
        print(f"Top {len(top_col_peaks)} column peaks at indices: {top_col_peaks}")
    
    return top_row_peaks, top_col_peaks


def segment_sudoku(image_path, visualize = False):
    # Load the image
    img = cv2.imread(image_path)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply adaptive thresholding to get binary image
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY_INV, 11, 2)
    
    # Dilate to fill gaps in lines
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find the largest contour, which should be the Sudoku grid
    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Approximate the contour to a polygon
        epsilon = 0.1 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        # If we have a quadrilateral, we can perform perspective transform
        if len(approx) == 4:
            # Order the points in the correct order (top-left, top-right, bottom-right, bottom-left)
            pts = np.array([approx[0][0], approx[1][0], approx[2][0], approx[3][0]], dtype="float32")
            # Order the points
            rect = order_points(pts)
            
            # Get width and height of the Sudoku grid
            (tl, tr, br, bl) = rect
            widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
            widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
            heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
            heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
            
            # Take maximum width and height
            maxWidth = max(int(widthA), int(widthB))
            maxHeight = max(int(heightA), int(heightB))
            
            # Destination points for perspective transform
            dst = np.array([
                [0, 0],
                [maxWidth - 1, 0],
                [maxWidth - 1, maxHeight - 1],
                [0, maxHeight - 1]
            ], dtype="float32")
            
            # Calculate perspective transform matrix and apply it
            M = cv2.getPerspectiveTransform(rect, dst)
            warped = cv2.warpPerspective(gray, M, (maxWidth, maxHeight))

            # Get the dimensions of the image
            height, width = warped.shape

            blurred2 = cv2.GaussianBlur(warped, (5, 5), 0)
            edges = cv2.adaptiveThreshold(blurred2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY_INV, 11, 2)
            
            # dilated = cv2.dilate(edges, kernel, iterations=1)
            
            # Apply Hough Line Transform to detect grid lines
            # edges = cv2.Canny(warped, 50, 150, apertureSize=3)
            # lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
            
            row_line, col_line = image_histogram_analysis(edges, visualize)



            # Create a copy of the warped image to draw lines on
            line_image = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)

            if visualize:
                # Display the results
                plt.figure(figsize=(12, 4))
                plt.subplot(131)
                plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                plt.title('Original Image')
                plt.axis('off')
                
                plt.subplot(132)
                plt.imshow(warped, cmap='gray')
                plt.title('Perspective Transform')
                plt.axis('off')
                
                plt.subplot(133)
                plt.imshow(cv2.cvtColor(line_image, cv2.COLOR_BGR2RGB))
                plt.title('Detected Grid Lines')
                plt.axis('off')
                
                plt.tight_layout()
                plt.show()


            
            for row in row_line:
                color = (255, 0, 0)
                start_point = (0, row)
                end_point = (width, row)
                cv2.line(line_image, start_point, end_point, color, 3)

            for col in col_line:
                color = (255, 0, 0)
                start_point = (col, 0)
                end_point = (col, height)
                cv2.line(line_image, start_point, end_point, color, 3)
            
            out = []
            row_line.sort()
            col_line.sort()
            

            prev_row = row_line[0]
            for i, row in enumerate(row_line[1:]):
                prev_col = col_line[0]
                out.append([])
                for col in col_line[1:]:
                    cell = warped[prev_row:row,prev_col:col]
                    out[i].append(cell)
                    prev_col = col
                prev_row = row

            return out
        



    return img, None, None

def order_points(pts):
    # Initialize a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    
    # The top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    # Now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    # Return the ordered coordinates
    return rect


def display_array_grid(arrays_9x9):
    """
    Display a 9x9 grid of numpy arrays as images.
    
    Args:
        arrays_9x9: A 9x9 list where each element is a numpy array
                    (all arrays should have the same shape)
    """
    # Verify input dimensions
    if len(arrays_9x9) != 9 or any(len(row) != 9 for row in arrays_9x9):
        raise ValueError("Input must be a 9x9 list of arrays")
    
    # Create a figure with a grid of subplots
    fig = plt.figure(figsize=(15, 15))
    gs = GridSpec(9, 9, figure=fig)
    
    # Get the global min and max values for consistent color scaling
    all_arrays = [array for row in arrays_9x9 for array in row]
    vmin = min(array.min() for array in all_arrays)
    vmax = max(array.max() for array in all_arrays)
    
    # Plot each array in its position
    for i in range(9):
        for j in range(9):
            ax = fig.add_subplot(gs[i, j])
            im = ax.imshow(arrays_9x9[i][j], cmap='viridis', vmin=vmin, vmax=vmax)
            
            # Remove ticks and axis labels for cleaner appearance
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Optional: Add subtle grid lines
            ax.spines['top'].set_visible(True)
            ax.spines['right'].set_visible(True)
            ax.spines['bottom'].set_visible(True)
            ax.spines['left'].set_visible(True)
    
    # Add a colorbar for reference
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    fig.colorbar(im, cax=cbar_ax)
    
    # Adjust spacing between subplots
    plt.tight_layout()
    plt.subplots_adjust(right=0.9)  # Make room for colorbar
    
    plt.show()
    
    return fig

def main():
    # Replace with your image path
    image_path = "488368_1_En_55_Fig1_HTML.png"
    
    # Segment the Sudoku grid
    grid = segment_sudoku(image_path)

    display_array_grid(grid)
    

if __name__ == "__main__":
    main()