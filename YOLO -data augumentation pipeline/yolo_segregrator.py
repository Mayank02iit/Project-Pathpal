import os
import shutil
import psutil

if not os.path.exists("all_data_3"):
    os.makedirs("all_data_3/images")
    os.makedirs("all_data_3/labels")
count = 0

for f in os.listdir("new_data_2"):
    image_files = sorted(os.listdir(f"new_data_2/{f}/images"))  # Sorting to match images and labels correctly
    label_files = sorted(os.listdir(f"new_data_2/{f}/labels"))  # Sorting ensures correct correspondence

    for image, label in zip(image_files, label_files):
        image_path = f"new_data_2/{f}/images/{image}"
        label_path = f"new_data_2/{f}/labels/{label}"

        # Copy label file
        with open(label_path, 'r') as file1, open(f"all_data_3/labels/{count}.txt", 'w') as file2:
            shutil.copyfileobj(file1, file2)

        # Move image file
        shutil.move(image_path, f"all_data_3/images/{count}.jpg")  # Moves instead of copying

        count += 1
# img_path = r"C:\Users\Mayank\Desktop\augumentation_data\all_data_3\images"
# label_path = r"C:\Users\Mayank\Desktop\augumentation_data\all_data_3\labels"
# for image , label in zip(os.listdir(img_path),os.listdir(label_path)):
#     total_img_path= os.path.join(img_path,image)
#     total_label_path = os.path.join(label_path,label)
#     with open(total_label_path,"r") as f:
#         data = f.readlines()
#         print(data)
#         if(len(data)==0):
#             print(total_img_path)
#             os.remove(total_img_path)
#                 # with open(total_label_path,"w") as f:
#                 #     pass