import os
import shutil
import glob

# Paths
SRC_BASE = r"c:\fun_project\yazlab_3\data\data_that_need_to_be_placed_in_their_adress"
DEST_BASE = r"c:\fun_project\yazlab_3\data\raw"

# Class mappings
# Source folder name -> Target raw folder name
CARS_BODY_TYPE_MAPPING = {
    "SUV": "SUV",
    "VAN": "VAN",
    "Sedan": "SEDAN",
    "Hatchback": "HATCHBACK",
    "Pick-Up": "PICK_UP"
}

def move_cars_body_type():
    src_dir = os.path.join(SRC_BASE, "Cars_Body_Type")
    if not os.path.exists(src_dir):
        print(f"Not found: {src_dir}")
        return

    # Cars_Body_Type has 'train', 'test', 'valid'
    for split in ["train", "test", "valid"]:
        split_dir = os.path.join(src_dir, split)
        if not os.path.exists(split_dir):
            continue
            
        for src_folder, target_folder in CARS_BODY_TYPE_MAPPING.items():
            current_src = os.path.join(split_dir, src_folder)
            current_dest = os.path.join(DEST_BASE, target_folder)
            
            if os.path.exists(current_src):
                images = glob.glob(os.path.join(current_src, "*.*"))
                for img in images:
                    filename = f"{split}_{os.path.basename(img)}"
                    shutil.move(img, os.path.join(current_dest, filename))
                print(f"Moved {len(images)} images from {src_folder} ({split}) to {target_folder}")

def move_f1_cars():
    src_dir = os.path.join(SRC_BASE, "F1_new_upd_car_data")
    if not os.path.exists(src_dir):
        print(f"Not found: {src_dir}")
        return
        
    target_dest = os.path.join(DEST_BASE, "F1")
    os.makedirs(target_dest, exist_ok=True)
    
    # F1 has subdirectories for each team
    for team_dir in os.listdir(src_dir):
        team_path = os.path.join(src_dir, team_dir)
        if os.path.isdir(team_path):
            images = glob.glob(os.path.join(team_path, "*.*"))
            for img in images:
                filename = f"{team_dir.replace(' ', '_')}_{os.path.basename(img)}"
                shutil.move(img, os.path.join(target_dest, filename))
            print(f"Moved {len(images)} images from {team_dir} to F1")

if __name__ == "__main__":
    print("Moving files to raw data directories...")
    move_cars_body_type()
    move_f1_cars()
    print("Moving complete!")
