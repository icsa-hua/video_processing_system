
import os
import pdb
import shutil 
import random
from tqdm import tqdm 

COCO_IMAGE_PATH = "../datasets/coco/images" 
COCO_LABELS_PATH = "../datasets/coco/labels"
OUTPUT_IMAGE_PATH = "assets/unified_dataset/images"
OUTPUT_LABEL_PATH = "assets/unified_dataset/labels" 

coco_train_path_images = os.path.join(COCO_IMAGE_PATH, "train2017")
coco_train_path_labels = os.path.join(COCO_LABELS_PATH, "train2017")
coco_train_output_images = os.path.join(OUTPUT_IMAGE_PATH, "train")
coco_train_output_labels = os.path.join(OUTPUT_LABEL_PATH, "train")

coco_val_path_images = os.path.join(COCO_IMAGE_PATH, "val2017")
coco_val_path_labels = os.path.join(COCO_LABELS_PATH, "val2017")
coco_val_output_images = os.path.join(OUTPUT_IMAGE_PATH, "val")
coco_val_output_labels = os.path.join(OUTPUT_LABEL_PATH, "val")


coco_test_path_images = os.path.join(COCO_IMAGE_PATH, "test2017") 
coco_test_path_labels = os.path.join(COCO_LABELS_PATH, "test2017") 
coco_test_output_images = os.path.join(OUTPUT_IMAGE_PATH, "test")
coco_test_output_labels = os.path.join(OUTPUT_LABEL_PATH, "test")

SAMPLE_SIZE = 3000 
VAL_SAMPLE = 1000 
TEST_SAMPLE = 3000

def real_labels_images(label_path, image_path, output_label_path, output_image_path, sample_size): 
    all_labels_files = [f for f in os.listdir(label_path) if f.endswith('.txt')] 
    if len(all_labels_files) < sample_size: 
        print(f"WARNING: ONLY {len(all_labels_files)} are available") 
        sample_files = all_labels_files 

    else: 
        sample_files = random.sample(all_labels_files, sample_size) 

    print(f"STARTING to copy {len(sample_files)} image-label pairs")

    copied_count = 0 

    for label_filename in tqdm(sample_files): 
        base_filename = os.path.splitext(label_filename)[0] 

        image_filename = base_filename + '.jpg'

        src_label_path = os.path.join(label_path, label_filename) 
        trg_label_path = os.path.join(output_label_path, label_filename)

        src_image_path = os.path.join(image_path, image_filename) 
        trg_image_path = os.path.join(output_image_path, image_filename) 

        if not os.path.exists(src_image_path): 
            print(f"ERROR: Image {image_filename} not found, skipping label {label_filename}")
            continue

        shutil.copy2(src_label_path, trg_label_path) 
        shutil.copy2(src_image_path, trg_image_path)
        
        copied_count += 1 
        if copied_count % 1000 == 0 : 
            print(f"Copied {copied_count} files ... ")

    print(f"\n✅ Finished. Successfully copied {copied_count} image-label pairs for COCO rehearsal.")


def transfer_images_only(image_path, output_image_path, sample_size): 
    all_image_files = [f for f in os.listdir(image_path) if f.endswith('.jpg') ]
    
    if len(all_image_files) < sample_size: 
        sample_files = all_image_files 

    else: 
        sample_files = random.sample(all_image_files, sample_size)

    print(f"STARTING to copy {len(sample_files)} image-label pairs")

    copied_count = 0 
    for image_filename in tqdm(sample_files): 
        src_image_path = os.path.join(image_path, image_filename) 
        trg_image_path = os.path.join(output_image_path, image_filename)

        if not os.path.exists(src_image_path): 
            raise ValueError(f"ERROR: Image {image_filename} not found")

        shutil.copy2(src_image_path, trg_image_path)

        copied_count += 1 
        if copied_count % 1000 == 0 : 
            print(f"Coppied {copied_count} files...")

    print(f"\n✅ Finished. Successfully copied {copied_count} image-label pairs for COCO rehearsal.")




real_labels_images(
    label_path = coco_train_path_labels, 
    image_path = coco_train_path_images, 
    output_label_path = coco_train_output_labels, 
    output_image_path = coco_train_output_images, 
    sample_size=SAMPLE_SIZE
)
real_labels_images(
    label_path = coco_val_path_labels, 
    image_path = coco_val_path_images, 
    output_label_path = coco_val_output_labels, 
    output_image_path = coco_val_output_images, 
    sample_size=VAL_SAMPLE
)


# transfer_images_only(
#     image_path=coco_test_path_images, 
#     output_image_path=coco_test_output_images, 
#     sample_size=TEST_SAMPLE
# )
