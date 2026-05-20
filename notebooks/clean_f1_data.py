import os
import glob
import numpy as np
from PIL import Image, ImageFilter

TARGET_COUNT = 970
F1_DIR = r"c:\fun_project\yazlab_3\data\raw\F1"

def compute_image_quality(img_path):
    try:
        # Load image in Grayscale
        img = Image.open(img_path).convert("L")
        arr = np.array(img)
        
        # 1. Blurriness / Sharpness
        # Apply a simple edge filter. The variance of the edges indicates sharpness.
        edges = img.filter(ImageFilter.FIND_EDGES)
        sharpness = np.var(np.array(edges))
        
        # 2. Contrast
        contrast = np.std(arr)
        
        # 3. Brightness (Check for overexposed or pitch black images)
        brightness = np.mean(arr)
        # We penalize extreme brightness or extreme darkness
        # Ideal brightness is around 127
        brightness_penalty = abs(brightness - 127)
        
        # Combine into a single quality score
        # High sharpness is good, high contrast is good, extreme brightness is bad
        # We add 1 to avoid division by zero
        score = (sharpness * contrast) / (brightness_penalty + 1)
        
        return score
    except Exception as e:
        # If the image is corrupted, give it the lowest possible score
        print(f"Error reading {img_path}: {e}")
        return -1

def clean_f1_dataset():
    if not os.path.exists(F1_DIR):
        print(f"Directory not found: {F1_DIR}")
        return

    images = glob.glob(os.path.join(F1_DIR, "*.*"))
    current_count = len(images)
    
    if current_count <= TARGET_COUNT:
        print(f"F1 class already has {current_count} images, which is <= target ({TARGET_COUNT}). No deletion needed.")
        return
        
    num_to_delete = current_count - TARGET_COUNT
    print(f"Found {current_count} F1 images. Target is {TARGET_COUNT}.")
    print(f"We need to delete the {num_to_delete} lowest-quality (blurry, bad lighting, corrupted) images.")
    
    # Calculate scores for all images
    print("Analyzing image qualities... This may take a minute.")
    image_scores = []
    
    for i, img_path in enumerate(images):
        score = compute_image_quality(img_path)
        image_scores.append((score, img_path))
        if (i + 1) % 100 == 0:
            print(f"  Analyzed {i + 1}/{current_count} images...")
            
    # Sort images by score in ascending order (lowest quality first)
    image_scores.sort(key=lambda x: x[0])
    
    # Delete the lowest scoring images
    print(f"\nDeleting the {num_to_delete} lowest quality images...")
    for i in range(num_to_delete):
        score, img_path = image_scores[i]
        try:
            os.remove(img_path)
            # print(f"Deleted {os.path.basename(img_path)} (Score: {score:.2f})")
        except Exception as e:
            print(f"Could not delete {img_path}: {e}")
            
    # Verification
    remaining_count = len(glob.glob(os.path.join(F1_DIR, "*.*")))
    print(f"\nCleanup complete! F1 dataset now has {remaining_count} high-quality images.")

if __name__ == "__main__":
    clean_f1_dataset()
