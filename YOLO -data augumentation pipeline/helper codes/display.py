import os
import matplotlib.pyplot as plt
import numpy as np
import yaml
from PIL import Image
# def display_sample(img_path,label_path):

#     img = Image.open(img_path)

#     labels=[]

#     with open(label_path,"r") as f:
#         labels = f.readlines()
    
#     fig, ax = plt.subplots(1, figsize=(5, 5))
    
#     # Display the image on the axis
#     ax.imshow(img)
    
#     # Loop through each bounding box in the label data and draw it
#     for box in labels:
#         box = box.strip().split(" ")
#         # Get bounding box coordinates (in relative format)
#         x_center = float(box[1])
#         y_center = float(box[2])
#         width = float(box[3])
#         height = float(box[4])
        
#         # Convert from relative to absolute pixel coordinates
#         image_width, image_height = img.size
#         x_min = (x_center - width / 2) * image_width
#         y_min = (y_center - height / 2) * image_height
#         x_max = (x_center + width / 2) * image_width
#         y_max = (y_center + height / 2) * image_height
        
#         # Draw the bounding box on the image using matplotlib's Rectangle
#         ax.add_patch(plt.Rectangle(
#             (x_min, y_min), x_max - x_min, y_max - y_min, 
#             linewidth=2, edgecolor='red', facecolor='none'
#         ))
#         ax.text(
#             x_min, y_min - 10,  # Position above the bounding box
#             box[0],  # Class name
#             color='red', fontsize=12, fontweight='bold'
#             # bbox=dict(facecolor='white', alpha=0.5, edgecolor='red', boxstyle='round,pad=0.5')
#         )

#     # Remove axis and display the image with bounding boxes
#     ax.axis('off')
#     plt.show()
# # new_data\Carrot_new\images\85_aug_2.jpg

# display_sample(r"C:\Users\Mayank\Desktop\augumentation_data\test\train\images\0.jpg",r"C:\Users\Mayank\Desktop\augumentation_data\test\train\labels\0.txt")

def count_files_in_directory(directory):
    total_files = 0
    for root, dirs, files in os.walk(directory):
        total_files += len(files)  # Add the number of files in the current directory
    return total_files

def display_multiple_samples(img_paths, label_paths):

    rows = count_files_in_directory(img_paths)
    fig, axes = plt.subplots(rows, 2,figsize=(15, 15))  # Create a grid of subplots
    
    # Flatten axes array for easy iteration
    axes = axes.flatten()

    for i, (img_path, label_path) in enumerate(zip(os.listdir(img_paths), os.listdir(label_paths))):
        # Open the image
        img = Image.open(os.path.join(img_paths,img_path))
        
        # Read label file
        labels = []
        print(os.path.join(label_paths,label_path))
        with open(os.path.join(label_paths,label_path), "r") as f:
            labels = f.readlines()
        
        ax = axes[i]  # Select subplot
        ax.imshow(img)
        
        # Draw bounding boxes for each label
        for box in labels:
            box = box.strip().split(" ")
            x_center = float(box[1])
            y_center = float(box[2])
            width = float(box[3])
            height = float(box[4])
            
            # Convert from relative to absolute pixel coordinates
            image_width, image_height = img.size
            x_min = (x_center - width / 2) * image_width
            y_min = (y_center - height / 2) * image_height
            x_max = (x_center + width / 2) * image_width
            y_max = (y_center + height / 2) * image_height
            
            # Draw the bounding box on the image using matplotlib's Rectangle
            ax.add_patch(plt.Rectangle(
                (x_min, y_min), x_max - x_min, y_max - y_min,
                linewidth=2, edgecolor='red', facecolor='none'
            ))
            ax.text(
                x_min, y_min - 10,  # Position above the bounding box
                box[0],  # Class name
                color='red', fontsize=12, fontweight='bold'
            )

        ax.axis('off')  # Remove axis for clarity

    plt.tight_layout()  # Adjust subplots for better spacing
    plt.show()

display_multiple_samples(r"C:\Users\Mayank\Desktop\augumentation_data\new_data_2\Orange_new\images",r"C:\Users\Mayank\Desktop\augumentation_data\new_data_2\Banana_new\labels")