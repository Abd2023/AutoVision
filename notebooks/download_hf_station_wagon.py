import os
from datasets import load_dataset

# Configuration
DATASET_NAME = "DamianBoborzi/car_images"
OUTPUT_DIR = "../data/raw/STATION_WAGON"
TARGET_COUNT = 970

# Keywords indicating a 'Station Wagon' car
KEYWORDS = [
    "station wagon", "estate car", "wagon", "touring", "avant", "variant",
    "combi", "volvo v60", "volvo v90", "audi a4 avant", "audi a6 avant",
    "passat estate", "octavia estate", "subaru outback"
]

def is_station_wagon(text, filename):
    text_lower = str(text).lower()
    filename_lower = str(filename).lower()
    
    for kw in KEYWORDS:
        if kw in text_lower or kw in filename_lower:
            return True
    return False

def download_wagon_dataset():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Connecting to HuggingFace to stream {DATASET_NAME} for Station Wagons...")
    
    # We use streaming=True
    streaming_data = load_dataset(DATASET_NAME, split="train", streaming=True)
    
    downloaded_count = 0
    scanned_count = 0
    
    print(f"Scanning for Station Wagon cars (Target: {TARGET_COUNT})...")
    
    for item in streaming_data:
        scanned_count += 1
        
        # Check if the text or filename contains our keywords
        if is_station_wagon(item.get("text", ""), item.get("original_filename", "")):
            img = item["image"]
            
            save_path = os.path.join(OUTPUT_DIR, f"wagon_{downloaded_count:04d}.jpg")
            
            if img.mode != "RGB":
                img = img.convert("RGB")
                
            img.save(save_path, "JPEG", quality=95)
            
            downloaded_count += 1
            if downloaded_count % 50 == 0:
                print(f"  [{downloaded_count}/{TARGET_COUNT}] downloaded. (Scanned {scanned_count} images)")
                
            if downloaded_count >= TARGET_COUNT:
                print("Target count reached! Stopping stream.")
                break
                
    print(f"\nDone! Successfully saved {downloaded_count} Station Wagons to {OUTPUT_DIR}")

if __name__ == "__main__":
    download_wagon_dataset()
