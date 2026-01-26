import os
from PIL import Image

def convert_png_to_jpeg(input_dir, output_dir):
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Iterate over all files in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith(".png"):
            # Construct full file paths
            png_path = os.path.join(input_dir, filename)
            jpeg_filename = os.path.splitext(filename)[0] + ".jpeg"
            jpeg_path = os.path.join(output_dir, jpeg_filename)

            # Open the PNG image and convert to JPEG
            with Image.open(png_path) as img:
                img = img.convert("RGB")
                img.save(jpeg_path, "JPEG")

# Example usage
input_directory = "/mnt/c/Users/DGeorgiadis_HUA/Downloads/Fisheye8K/FishEye8K/FishEye8K_splits/train/images"
output_directory = "assets/unified_dataset/images/train"
convert_png_to_jpeg(input_directory, output_directory)

# Example usage
input_directory = "/mnt/c/Users/DGeorgiadis_HUA/Downloads/Fisheye8K/FishEye8K/FishEye8K_splits/val/images"
output_directory = "assets/unified_dataset/images/val"
convert_png_to_jpeg(input_directory, output_directory)