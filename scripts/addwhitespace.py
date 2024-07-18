#!/usr/bin/env python
import argparse
import os
from PIL import Image

def create_wider_image(input_path, output_path, factor=2.0):
    # Open the original image
    original_image = Image.open(input_path)
    original_width, original_height = original_image.size
    
    # Define the new dimensions
    new_width = int(factor * original_width)
    new_height = original_height
    
    # Create a new image with a white background
    new_image = Image.new("RGBA", (new_width, new_height), (255, 255, 255, 0))
    
    # Calculate the position to paste the original image
    paste_position = (int(original_width * (factor-1) / 2), 0)

    # Paste the original image onto the new image
    new_image.paste(original_image, paste_position)
    
    # Save the new image
    new_image.save(output_path)

def process_images(input_files, output_dir, factor=2.0):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for input_file in input_files:
        input_filename = os.path.basename(input_file)
        output_path = os.path.join(output_dir, input_filename)
        create_wider_image(input_file, output_path, factor=factor)
        print(f"Processed {input_file} -> {output_path}")

if __name__ == "__main__":
    home_dir = os.path.expanduser("~")
    default_output_dir = os.path.join(home_dir, "Desktop")

    parser = argparse.ArgumentParser(description="Create a wider image with the original centered.")
    parser.add_argument("input_files", nargs='+', help="Paths to the input image files.")
    parser.add_argument("--outdir", default=default_output_dir, help="Directory where the output images will be saved (default: $HOME/Desktop/)")
    parser.add_argument("--factor", type=float, default=2.0)
    
    args = parser.parse_args()
    process_images(args.input_files, args.outdir, factor=args.factor)