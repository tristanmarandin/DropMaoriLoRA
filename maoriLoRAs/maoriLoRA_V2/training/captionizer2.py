import os
import pandas as pd

# Specify the path to the Excel file
excel_path = r'C:/Users/Tristan/Documents/Drop/DatabaseLoRA/maoriLoRA/training/image_captions.xlsx'

# Specify the directory where text files will be saved
txt_dir = r'C:/Users/Tristan/Documents/Drop/DatabaseLoRA/maoriLoRA/training/img/txt_captions'

# Ensure that the directory exists
os.makedirs(txt_dir, exist_ok=True)

# Read the Excel file into a DataFrame
df = pd.read_excel(excel_path, engine='openpyxl')

# Iterate over the rows of the DataFrame, creating a text file for each
for index, row in df.iterrows():
    img_name, img_caption = row['image name'], row['image caption']
    
    # Ensure the caption is a string and not nan from pandas (which is a float)
    if isinstance(img_caption, float):
        print(f"Warning: Missing caption for {img_name}. Text file will be empty.")
        img_caption = ""
    
    # Remove .jpg from the image name
    img_name_no_ext = os.path.splitext(img_name)[0]
    
    # Create a .txt file for each image name, writing the corresponding caption inside
    with open(os.path.join(txt_dir, f"{img_name_no_ext}.txt"), 'w', encoding='utf-8') as txt_file:
        txt_file.write(img_caption)

print(f"Text files created in {txt_dir}.")
