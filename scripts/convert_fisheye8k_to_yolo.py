import os 
from tqdm import tqdm
import pdb

FISHEYE_LABELS_PATH = '/mnt/c/Users/DGeorgiadis_HUA/Downloads/Fisheye8K/FishEye8K/FishEye8K_splits/labels' 
OUTPUT_LABELS_PATH = 'assets/unified_dataset/labels' 
os.makedirs(OUTPUT_LABELS_PATH, exist_ok=True) 

ID_REMAP = {
    0: 5,  # Bus (FishEye8K) -> Bus (COCO)
    1: 1,  # Bike (FishEye8K) -> Bicycle (COCO)
    2: 2,  # Car (FishEye8K) -> Car (COCO)
    3: 0,  # Pedestrian (FishEye8K) -> Person (COCO)
    4: 7   # Truck (FishEye8K) -> Truck (COCO)
}

def remap_labels_in_directory(source_dir, dest_dir): 

    os.makedirs(dest_dir, exist_ok=True) 

    label_files = [f for f in os.listdir(source_dir) if f.endswith('.txt')] 
    
    for filename in tqdm(label_files): 
        input_path = os.path.join(source_dir, filename) 
        output_path = os.path.join(dest_dir, filename) 

        new_lines = [] 

        try: 
            with open(input_path, 'r') as f: 
                for line in f: 
                    parts = line.strip().split() 
                    if not parts: continue

                    old_class_id = int(parts[0]) 
                    coords = parts[1:] 

                    if old_class_id in ID_REMAP: 
                        new_class_id = ID_REMAP[old_class_id] 
                        new_line = f"{new_class_id} " + " ".join(coords) 
                        new_lines.append(new_line) 
                    else: 
                        print(f"WARNING:Skipping unknown class ID {old_class_id} in file {filename}")

            if new_lines: 
                with open(output_path, 'w') as f: 
                    f.write('\n'.join(new_lines) + '\n') 

        except Exception as e: 
            print(f"Error processing file {filename}:{e}")


def process_labels(name): 
    fisheye_source = os.path.join(FISHEYE_LABELS_PATH, name) 
    unified_dest = os.path.join(OUTPUT_LABELS_PATH,name ) 
    remap_labels_in_directory(fisheye_source, unified_dest) 



process_labels('train') 
process_labels('val')
print("\n--- Post-script Instructions ---")
print("1. CLASS REMAPPING COMPLETE.")
print(f"   Labels remapped and saved to: {OUTPUT_LABELS_PATH}")
print("2. NEXT STEP: Image Copying")
print("   You MUST manually copy the FishEye8K images (train/val) into 'unified_dataset/images/train' and 'unified_dataset/images/val'.")
print("3. FINAL STEP: COCO Rehearsal")
print("   Copy a sample of COCO images and their YOLO labels into the same 'unified_dataset/images/' and 'unified_dataset/labels/' subfolders.")
