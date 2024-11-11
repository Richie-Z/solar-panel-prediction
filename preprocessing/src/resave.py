import os
from openpyxl import load_workbook

data_folder = '/Users/richie/Downloads/Data/months_processed'

for filename in os.listdir(data_folder):
    if filename.endswith('.xlsx'):
        file_path = os.path.join(data_folder, filename)

        try:
            workbook = load_workbook(filename=file_path)
            workbook.save(filename=file_path)
            print(f"{filename} has been resaved successfully.")
        except Exception as e:
            print(f"Skipping {filename}: An error occurred - {e}")
