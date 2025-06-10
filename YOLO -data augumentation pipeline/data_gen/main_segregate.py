import os
import matplotlib.pyplot as plt
import numpy as np
import yaml
from PIL import Image
import shutil

dataset_path = r"C:\Users\Mayank\Desktop\augumentation_data\dataset"
train_path = os.path.join(dataset_path,'train')
val_path = os.path.join(dataset_path,'valid')
test_path = os.path.join(dataset_path,'test')
train_images_path = os.path.join(dataset_path,'train\images')
train_labels_path = os.path.join(dataset_path,'train\labels')
val_images_path = os.path.join(dataset_path,'valid\labels')
val_labels_path = os.path.join(dataset_path,'valid\labels')
test_images_path = os.path.join(dataset_path,'test\images')
test_labels_path = os.path.join(dataset_path,'test\labels')
base_path = r"C:\Users\Mayank\Desktop\augumentation_data"

# with open("object_names.txt",'r') as f:
#         objs = f.read().strip().split(' ')
objs = ['Apple', 'Banana', 'Bell Pepper', 'Bread', 'Carrot', 'Detergent', 'Drinks', 'Egg', 'Lemon', 'Orange', 'Strawberry']
print(os.path.exists(dataset_path))
def process_data(images_path, labels_path):

    image_files = [f for f in os.listdir(images_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    label_files = []

    for image_file in image_files:
        label_file = os.path.join(labels_path, image_file.replace(os.path.splitext(image_file)[1], '.txt'))
        if os.path.exists(label_file):
            label_files.append(label_file)
        else:
            print(f"Missing label for image: {image_file}")
            pass
    
    for image_file, label_file in zip(image_files, label_files):
        image_path = os.path.join(images_path, image_file)
        
        with open(label_file, 'r') as f:
            lines = f.readlines()

        boxes = []
        count_item = {}

        for line in lines:
            parts = line.strip().split()
            class_index = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            count_item[class_index] = count_item.get(class_index, 0) + 1
            boxes.append({
                'class': class_index,
                'x_center': x_center,
                'y_center': y_center,
                'width': width,
                'height': height
            })

        # Find class with the maximum count
        max_count = max(count_item.values())

        for obj, count in count_item.items():
            if count == max_count:
                obj_labels = [box for box in boxes if box['class'] == obj]

                # Paths for labels and images
                base_obj_label_path = os.path.join(base_path, objs[obj], "labels")
                base_obj_image_path = os.path.join(base_path, objs[obj], "images")

                os.makedirs(base_obj_label_path, exist_ok=True)
                os.makedirs(base_obj_image_path, exist_ok=True)

                # Write labels to the new file
                num_label_files = len([f for f in os.listdir(base_obj_label_path) if os.path.isfile(os.path.join(base_obj_label_path, f))])
                label_file_path = os.path.join(base_obj_label_path, f"{num_label_files}.txt")

                with open(label_file_path, 'w') as f:
                    for row in obj_labels:
                        f.write(f"{row['class']} {row['x_center']} {row['y_center']} {row['width']} {row['height']}\n")

                # Move and rename the image file
                num_image_files = len([f for f in os.listdir(base_obj_image_path) if os.path.isfile(os.path.join(base_obj_image_path, f))])
                new_image_name = f"{num_image_files}{os.path.splitext(image_file)[1]}"
                destination_image_path = os.path.join(base_obj_image_path, new_image_name)

                shutil.copy(image_path, destination_image_path)
                print(f"Moved and renamed image to: {destination_image_path}")

# process_data(train_images_path ,train_labels_path)
process_data(test_images_path ,test_labels_path)
process_data(val_images_path ,val_labels_path)
