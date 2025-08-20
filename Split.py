import os
import shutil
import random
import cv2

# Paths
dataset_path = r"C:\Users\Bhava\Downloads\archive"  # Folder containing 'male' and 'female'
output_path = r"C:\Users\Bhava\Desktop\Pranava\data"   # Will contain train/ and test/

# Train/Test split ratio
split_ratio = 0.8

# Clear old processed dataset if exists
if os.path.exists(output_path):
    shutil.rmtree(output_path)
os.makedirs(output_path)

# Create train/test folders
train_dir = os.path.join(output_path, "train")
test_dir = os.path.join(output_path, "test")
os.makedirs(train_dir)
os.makedirs(test_dir)

# Process each class folder (male/female)
for class_name in os.listdir(dataset_path):
    class_folder = os.path.join(dataset_path, class_name)
    if not os.path.isdir(class_folder):
        continue

    # Make class folders inside train/test
    os.makedirs(os.path.join(train_dir, class_name))
    os.makedirs(os.path.join(test_dir, class_name))

    # Get all images and shuffle
    images = os.listdir(class_folder)
    random.shuffle(images)

    split_index = int(len(images) * split_ratio)
    train_images = images[:split_index]
    test_images = images[split_index:]

    # Function to preprocess and save
    def process_and_save(image_list, save_folder):
        for img_name in image_list:
            img_path = os.path.join(class_folder, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            # Preprocessing
            img_enhanced = cv2.equalizeHist(img)
            img_edges = cv2.Canny(img_enhanced, 50, 150)
            save_path = os.path.join(save_folder, img_name)
            cv2.imwrite(save_path, img_edges)

    # Process train and test sets
    process_and_save(train_images, os.path.join(train_dir, class_name))
    process_and_save(test_images, os.path.join(test_dir, class_name))

print("✅ Dataset split & preprocessing done!")
print(f"Train folder: {train_dir}")
print(f"Test folder: {test_dir}")