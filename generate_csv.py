import csv
import os
import random

# Define the dataset directory
dataset_dir = "/igd/a1/home/ozgur/data/FaceMAD/evaluate_external/frgc_aligned/"  # Replace with your dataset directory
output_dir = "/igd/a1/home/ozgur/data/FaceMAD/evaluate_external/frgc_aligned/Protocols/"  # Replace with the directory to save CSVs

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# List of all subfolders (attack types) except 'bonafide'
all_folders = [
    folder
    for folder in os.listdir(dataset_dir)
    if os.path.isdir(os.path.join(dataset_dir, folder))
]
attack_folders = [folder for folder in all_folders if folder.lower() != "bonafide"]

# Collect bonafide images
bonafide_dir = os.path.join(dataset_dir, "bonafide")
bonafide_images = []
if os.path.exists(bonafide_dir):
    for root, _, files in os.walk(bonafide_dir):
        for file_name in files:
            if file_name.endswith((".png", ".jpg", ".jpeg")):
                bonafide_images.append([os.path.join(root, file_name), "bonafide"])

# Collect all attack images
all_attack_images = []

# Process each attack folder
for attack in attack_folders:
    attack_dir = os.path.join(dataset_dir, attack)
    attack_images = []

    # Collect attack images
    for root, _, files in os.walk(attack_dir):
        for file_name in files:
            if file_name.endswith((".png", ".jpg", ".jpeg")):
                attack_images.append([os.path.join(root, file_name), "attack"])
                all_attack_images.append([os.path.join(root, file_name), "attack"])

    # Combine bonafide and attack images for each attack type
    combined_data = bonafide_images + attack_images

    # Shuffle the data
    random.shuffle(combined_data)

    # Save to a CSV file for each attack type
    output_csv = os.path.join(output_dir, f"{attack}.csv")
    with open(output_csv, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["image_path", "label"])  # Write header
        writer.writerows(combined_data)  # Write shuffled data

    print(f"CSV created: {output_csv}")

# Combine bonafide and all attack images
combined_all_data = bonafide_images + all_attack_images

# Shuffle the combined data
random.shuffle(combined_all_data)

# Save to a CSV file including all attack images
output_csv_all_attacks = os.path.join(output_dir, "all_attacks.csv")
with open(output_csv_all_attacks, mode="w", newline="") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["image_path", "label"])  # Write header
    writer.writerows(combined_all_data)  # Write shuffled data

print(f"CSV created: {output_csv_all_attacks}")
