import os
from datasets import load_dataset

# Configuration
DATASET_NAME = "DamianBoborzi/car_images"
OUTPUT_DIR = "../data/raw/MICRO"
TARGET_COUNT = 1500

# Keywords indicating a 'Micro' car
KEYWORDS = [
    "fiat 500", "smart", "fortwo", "forfour", "mini cooper", 
    "twingo", "aygo", "peugeot 108", "citroen c1", "chevrolet spark", 
    "hyundai i10", "kia picanto", "vw up", "volkswagen up", "toyota iq"
]

def is_micro_car(text, filename):
    text_lower = str(text).lower()
    filename_lower = str(filename).lower()
    
    for kw in KEYWORDS:
        if kw in text_lower or kw in filename_lower:
            return True
    return False

def download_micro_dataset():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Connecting to HuggingFace to stream {DATASET_NAME}...")
    
    # We use streaming=True to avoid downloading the massive 15GB file!
    streaming_data = load_dataset(DATASET_NAME, split="train", streaming=True)
    
    downloaded_count = 0
    scanned_count = 0
    
    print(f"Scanning for Micro cars (Target: {TARGET_COUNT}). This might take a few minutes...")
    
    for item in streaming_data:
        scanned_count += 1
        
        # Check if the text or filename contains our keywords
        if is_micro_car(item.get("text", ""), item.get("original_filename", "")):
            # The 'image' field is already a PIL Image object!
            img = item["image"]
            
            # Save it
            save_path = os.path.join(OUTPUT_DIR, f"micro_{downloaded_count:04d}.jpg")
            
            # Convert to RGB just in case there are transparent PNGs
            if img.mode != "RGB":
                img = img.convert("RGB")
                
            img.save(save_path, "JPEG", quality=95)
            
            downloaded_count += 1
            if downloaded_count % 50 == 0:
                print(f"  [{downloaded_count}/{TARGET_COUNT}] downloaded. (Scanned {scanned_count} total images)")
                
            if downloaded_count >= TARGET_COUNT:
                print("Target count reached! Stopping stream.")
                break
                
    print(f"\nDone! Successfully saved {downloaded_count} Micro cars to {OUTPUT_DIR}")

if __name__ == "__main__":
    download_micro_dataset()
