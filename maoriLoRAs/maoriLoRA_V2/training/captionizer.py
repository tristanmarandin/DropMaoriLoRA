import os
import pandas as pd

# Path to the folder containing images
img_folder_path = r'C:/Users/Tristan/Documents/Drop/DatabaseLoRA/maoriLoRA/training/img/25_maoridb maoridrawing'

# Retrieve all files from the folder in order
image_files = sorted(os.listdir(img_folder_path))

# Filter out non-image files (considering common image file extensions)
image_files = [f for f in image_files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff'))]

# Create a DataFrame with image names
df = pd.DataFrame(image_files, columns=['image name'])

# Add an empty column for image captions
df['image caption'] = ''

# Save the DataFrame to an Excel file
excel_path = r'C:/Users/Tristan/Documents/Drop/DatabaseLoRA/maoriLoRA/training/image_captions.xlsx'
df.to_excel(excel_path, index=False, engine='openpyxl')

print(f"Excel file created at {excel_path}. Please fill in the image captions.")