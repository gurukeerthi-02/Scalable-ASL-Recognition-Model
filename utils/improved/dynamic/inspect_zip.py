import zipfile
import os

zip_path = r"D:\FinalYearProject\Asl-Recog\utils\improved\dynamic\zip_files\asl_dynamic_a.mohamedaashiq_20260214.zip"

print(f"Inspecting: {zip_path}")
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    for file in zip_ref.namelist()[:20]: # Print first 20 files
        print(file)
