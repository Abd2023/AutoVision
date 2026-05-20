import os
import glob
import random
from pathlib import Path
from rembg import remove, new_session
from PIL import Image

# ==========================================
# CONFIGURATION
# ==========================================
RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
CLASSES = [
    "SUV", "VAN", "STATION_WAGON", "SEDAN", 
    "HATCHBACK", "PICK_UP", "F1", "MICRO"
]

# We want exactly equal numbers to balance the dataset.
TARGET_COUNT = 970  

def ensure_dirs():
    for cls in CLASSES:
        os.makedirs(os.path.join(RAW_DIR, cls), exist_ok=True)
        os.makedirs(os.path.join(PROCESSED_DIR, cls), exist_ok=True)

def process_image(input_path, output_path, session):
    try:
        # Load image
        input_img = Image.open(input_path).convert("RGBA")
        
        # Remove background using U2NetP
        output_img = remove(input_img, session=session)
        
        # Create a solid gray background (domain bias mitigation)
        background = Image.new("RGB", output_img.size, (200, 200, 200))
        
        # Paste the car onto the gray background
        # output_img.split()[3] is the alpha channel acting as a mask
        background.paste(output_img, mask=output_img.split()[3])
        
        # Resize to standard 224x224 to save space and processing time later
        background = background.resize((224, 224), Image.Resampling.LANCZOS)
        
        # Save as JPG
        background.save(output_path, format="JPEG", quality=95)
        return True
    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        return False

def preprocess_dataset():
    print("Phase 2: Data Preprocessing & Balancing")
    ensure_dirs()
    
    print("Initializing U2NetP session for rembg (4.7 MB ultra-light model)...")
    # Using 'u2netp' ensures we don't download the heavy 170MB standard model
    session = new_session("u2netp")
    
    for cls in CLASSES:
        cls_raw_dir = os.path.join(RAW_DIR, cls)
        cls_processed_dir = os.path.join(PROCESSED_DIR, cls)
        
        images = glob.glob(os.path.join(cls_raw_dir, "*.*"))
        
        if len(images) == 0:
            print(f"Warning: No raw images found for '{cls}'. Please place images in {cls_raw_dir}")
            continue
            
        print(f"Found {len(images)} images for {cls}.")
        
        # Shuffle and select EXACTLY TARGET_COUNT to ensure perfect balance
        random.shuffle(images)
        selected_images = images[:TARGET_COUNT]
        
        print(f"Processing {len(selected_images)} images for {cls} to ensure balance...")
        
        success_count = 0
        for i, img_path in enumerate(selected_images):
            filename = os.path.basename(img_path)
            out_path = os.path.join(cls_processed_dir, f"{os.path.splitext(filename)[0]}.jpg")
            
            # Skip if already processed
            if os.path.exists(out_path):
                success_count += 1
                continue
                
            if process_image(img_path, out_path, session):
                success_count += 1
                
            if (i+1) % 100 == 0:
                print(f"  [{i+1}/{len(selected_images)}] done.")
                
        print(f"Finished {cls}. Successfully processed: {success_count}/{len(selected_images)}\n")

if __name__ == '__main__':
    preprocess_dataset()
