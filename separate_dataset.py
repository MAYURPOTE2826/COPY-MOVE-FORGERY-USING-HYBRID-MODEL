import os
import shutil

# ---- PATHS ----
source_folder = "raw_dataset"
original_folder = "Dataset/original"
forged_folder = "Dataset/forged"

# ---- Create destination folders if not exist ----
os.makedirs(original_folder, exist_ok=True)
os.makedirs(forged_folder, exist_ok=True)

# ---- Separate files ----
for filename in os.listdir(source_folder):
    
    if "_O" in filename and "_F" not in filename:
        shutil.copy(
            os.path.join(source_folder, filename),
            os.path.join(original_folder, filename)
        )
    
    elif "_F" in filename:
        shutil.copy(
            os.path.join(source_folder, filename),
            os.path.join(forged_folder, filename)
        )

print("✅ Dataset separated successfully!")