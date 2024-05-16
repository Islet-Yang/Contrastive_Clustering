import os
import cv2
import numpy as np
from scipy.spatial import ConvexHull
from tqdm import tqdm

def selcet(directory, save_directory):
    qualified_num = 0
    total_num = 0
    for filename in tqdm(os.listdir(directory), desc="Processing images"):
        if filename.endswith(".png"):
            image_path = os.path.join(directory, filename)
            image = cv2.imread(image_path)
            total_num += 1
            if image_qualified(image):
                cv2.imwrite(os.path.join(save_directory, filename), image)
                qualified_num += 1

    print(f"Done. {qualified_num} images are selected. Percentage: {qualified_num / total_num:.2f}")               

def image_qualified(image):
    if image.shape != (60, 60, 3):
        return False
    
    # Define the thresholds
    gray_scale_threshold = 220
    quantity_threshold = 120
    cluster_threshold = 500

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    qualified_pixels = np.argwhere(gray_image > gray_scale_threshold)

    if len(qualified_pixels) > quantity_threshold:
        # Calculate the convex hull of the qualified pixels
        if len(qualified_pixels) >= 3:  # At least 3 points are needed to form a convex hull
            hull = ConvexHull(qualified_pixels)
            hull_area = hull.volume  # The area of the convex hull
            return hull_area < cluster_threshold
        else:
            return True
    else:
        return False
    

# Main function
if __name__ == "__main__":
    directory = "./newdata/"
    save_directory = "./Figures/"
    for filename in os.listdir(save_directory):
        os.remove(os.path.join(save_directory, filename))
    selcet(directory, save_directory)
