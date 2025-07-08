import os
import shutil
import random 
import sys
from PIL import Image 
# This script prepares the dataset for training, validation, and testing.
SOURCE_DIR = '../data/PetImages'
BASE_DIR = '../data'

def create_dir_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f'Created directory: {path}')

# load filenames into list
# shuffle 
# split into train 0.8, validation 0.1, test 0.1
# save into folder structure

train_dir = os.path.join(BASE_DIR, 'train')
validation_dir = os.path.join(BASE_DIR, 'validation')
test_dir = os.path.join(BASE_DIR, 'test')

create_dir_if_not_exists(BASE_DIR)
create_dir_if_not_exists(train_dir)
create_dir_if_not_exists(validation_dir)
create_dir_if_not_exists(test_dir)

classes = ['Dog', 'Cat']
valid_images = []

for class_name in classes:
    dir = os.path.join(SOURCE_DIR, class_name)
    # read files in the directory 
    images = [file for file in os.listdir(dir) if os.path.isfile(os.path.join(dir, file)) and file.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # filter out broken images
    for image in images:
        try:
            img_path = os.path.join(dir, image)
            # Check for zero-byte files
            if os.path.getsize(img_path) == 0:
                print(f"Skipping zero-byte file: {img_path}")
                continue
            try:
                Image.open(img_path).load()
            except:
                print(f"Skipping corrupt image: {img_path}")
                continue
            valid_images.append(img_path)
        except Exception as e:
            print(f"Error checking {img_path}: {e}. Skipping.")
            continue
    print(f'Found {len(valid_images)} valid images in {class_name} class.')

# Shuffle the valid images
random.shuffle(valid_images)
# Split the images into train, validation, and test sets
num_train = int(0.8 * len(valid_images))
num_validation = int(0.1 * len(valid_images))
num_test = len(valid_images) - num_train - num_validation

# Copy images to the respective directories 
for i, image_path in enumerate(valid_images):
    class_name = 'Dog' if 'Dog' in image_path else 'Cat'
    if i < num_train:
        target_dir = os.path.join(train_dir, class_name)
    elif i < num_train + num_validation:
        target_dir = os.path.join(validation_dir, class_name)
    else:
        target_dir = os.path.join(test_dir, class_name)
    create_dir_if_not_exists(target_dir)

    # Copy the image to the target directory
    shutil.copy(image_path, target_dir)

print(f'Train set: {num_train} images')
print(f'Validation set: {num_validation} images')
print(f'Test set: {num_test} images')
print('Data preparation complete.')
print(f'Images are saved in {BASE_DIR} directory.')


