

"""
This script processes images by removing their backgrounds and replacing them with a green screen.
It uses two background removal models, RMBG-1.4 and RMBG-2.0, to perform the segmentation.

Steps:
1. Loads images from a specified input folder.
2. Applies RMBG-1.4 and RMBG-2.0 models for background removal.
3. Replaces the transparent background with a green screen (RGB: 0, 255, 0).
4. Combines the original image, RMBG-1.4 output, and RMBG-2.0 output with green screen backgrounds into a single image.
5. Saves the final combined image to the specified output folder with titles for each version.

Dependencies:
- transformers
- PIL (Pillow)
- torch
- torchvision

Usage:
Specify the `input_folder_path` and `output_folder_path`, and run the script to generate images with a green screen background.
"""

import os
from transformers import pipeline, AutoModelForImageSegmentation
from PIL import Image, ImageDraw, ImageFont
import torch
from torchvision import transforms

# Paths to the image folders
input_folder_path = "rmbg/rmbg_benchmark/gen_ai_benchmark"
output_folder_path = "rmbg/rmbg_benchmark/gen_ai_benchmark_rmbg_results"

os.makedirs(output_folder_path, exist_ok=True)

# Initialize RMBG-1.4 pipeline
pipe_1_4 = pipeline("image-segmentation", model="briaai/RMBG-1.4", trust_remote_code=True)

# Load RMBG-2.0 model
birefnet = AutoModelForImageSegmentation.from_pretrained('briaai/RMBG-2.0', trust_remote_code=True)
birefnet.to('cuda').eval()

# Transformation for RMBG-2.0
transform_image = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Font settings for titles
font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
font_size = 30
font = ImageFont.truetype(font_path, font_size)

# Process each image in the input folder
for filename in os.listdir(input_folder_path):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(input_folder_path, filename)
        print(f"Processing {filename}...")

        # Load the original image
        original_image = Image.open(image_path)

        # Process with RMBG-1.4
        pillow_image_1_4 = pipe_1_4(image_path)  # Returns image with removed background

        # Add green screen to RMBG-1.4 result
        green_bg_1_4 = Image.new("RGBA", pillow_image_1_4.size, (0, 255, 0))
        green_bg_1_4.paste(pillow_image_1_4, (0, 0), pillow_image_1_4)

        # Process with RMBG-2.0
        input_image = transform_image(original_image).unsqueeze(0).to('cuda')
        with torch.no_grad():
            preds = birefnet(input_image)[-1].sigmoid().cpu()
        pred = preds[0].squeeze()

        mask_2_0 = transforms.ToPILImage()(pred).resize(original_image.size)
        image_rmbg_2 = original_image.copy()
        image_rmbg_2.putalpha(mask_2_0)

        # Add green screen to RMBG-2.0 result
        green_bg_2_0 = Image.new("RGBA", image_rmbg_2.size, (0, 255, 0))
        green_bg_2_0.paste(image_rmbg_2, (0, 0), image_rmbg_2)

        # Create a concatenated image with original, RMBG-1.4 with green screen, and RMBG-2.0 with green screen outputs
        combined_width = original_image.width + green_bg_1_4.width + green_bg_2_0.width
        combined_image = Image.new("RGBA", (combined_width, original_image.height + font_size + 10))
        
        # Draw titles
        draw = ImageDraw.Draw(combined_image)
        draw.text((original_image.width // 2 - font_size, 5), "Original Image", font=font, fill="white")
        draw.text((original_image.width + green_bg_1_4.width // 2 - font_size, 5), "RMBG 1.4", font=font, fill="white")
        draw.text((original_image.width + green_bg_1_4.width + green_bg_2_0.width // 2 - font_size, 5), "RMBG 2.0", font=font, fill="white")

        # Paste images below titles
        combined_image.paste(original_image, (0, font_size + 10))
        combined_image.paste(green_bg_1_4, (original_image.width, font_size + 10))
        combined_image.paste(green_bg_2_0, (original_image.width + green_bg_1_4.width, font_size + 10))

        # Save the final combined image
        output_file_path = os.path.join(output_folder_path, f"combined_{filename}")
        combined_image.save(output_file_path)

        print(f"Saved combined image to {output_file_path}")
