import albumentations as A
import cv2
import numpy as np
import os


def transform():
    return A.Compose([
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.MotionBlur(blur_limit=(3, 7), p=0.3),
        A.Perspective(scale=(0.05, 0.15), p=0.5, fit_output=False),  # Keep within bounds
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.ShiftScaleRotate(
            shift_limit=0.05, 
            scale_limit=0.1, 
            rotate_limit=30, 
            p=0.5, 
            border_mode=0,  # Ensure shifted pixels are within bounds
            value=0,        # Fill with black if shifting beyond borders
            keep_size=True  # Keeps output the same size as input
        ),
        A.GaussianBlur(blur_limit=(3, 5), p=0.3),
        A.GaussNoise(var_limit=(10.0, 30.0), p=0.3)
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))


def load_yolo_bboxes(txt_path):
    with open(txt_path, 'r') as f:
        lines = f.readlines()

    class_labels = []
    bounding_boxes = []

    for line in lines:
        line = line.strip().split(" ")
        class_label = int(line[0])
        bounding_box = np.array(list(map(float, line[1:])))
        class_labels.append(class_label)
        bounding_boxes.append(bounding_box)

    bounding_boxes  = np.array(bounding_boxes)
    print(bounding_boxes)
    if len(bounding_boxes)==1:
        bounding_boxes.squeeze(axis = 0)
    return bounding_boxes, class_labels

def save_yolo_bboxes(txt_path, bboxes, class_labels):
    """ Save transformed bounding boxes back to a YOLO format text file """
    with open(txt_path, "w") as f:
        for cls, bbox in zip(class_labels, bboxes):
            f.write(f"{cls} " + " ".join(map(str, bbox)) + "\n")


def augument_yolo_data(img_path, txt_path, output_dir,output_dir_label, img_name_suffix="_aug",num_aug = 1):
    """ Apply augmentation to an image and its YOLO annotation file """
    
    # Load image
    img = cv2.imread(img_path)

    # Load bounding boxes
    if os.path.exists(txt_path):
        print(1)
        bboxes, class_labels = load_yolo_bboxes(txt_path)
    else:
        print(1)
        bboxes, class_labels = [], []
        
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(output_dir_label):
        os.makedirs(output_dir_label)

    # Apply transformations
    for  i in range(0,num_aug):
      transform_obj = transform()

      augmented = transform_obj(image=img, bboxes=bboxes, class_labels=class_labels)

      aug_img = augmented["image"]
      aug_bboxes = augmented["bboxes"]
      aug_labels = augmented["class_labels"]

      # Save augmented image
    #   C:\Users\Mayank\Desktop\augumentation_data\Strawberry\images\93.jpg
      base_name = os.path.basename(img_path).split('.')[0]
      aug_img_name = f"{base_name}{img_name_suffix}_{i}.jpg"
      aug_txt_name = f"{base_name}{img_name_suffix}_{i}.txt"
    
      print(os.path.join(output_dir, aug_img_name))
      print(11111111111111111111)
      print(os.path.join(output_dir_label, aug_txt_name))
      cv2.imwrite(os.path.join(output_dir, aug_img_name), aug_img)

      # Save augmented YOLO annotation
      save_yolo_bboxes(os.path.join(output_dir_label, aug_txt_name), aug_bboxes, aug_labels)

# Example Usage
# img_path = "images/sample.jpg"
# txt_path = "labels/sample.txt"
# output_dir = "augmented_data"
# os.makedirs(output_dir, exist_ok=True)
# augment_yolo_data(img_path, txt_path, output_dir)
objs = ['Apple', 'Banana', 'Bell Pepper', 'Bread', 'Carrot', 'Detergent', 'Drinks', 'Egg', 'Lemon', 'Orange', 'Strawberry']
def generate_data(obj,num,instances):
    base_path = r"C:\Users\Mayank\Desktop\augumentation_data\no_aug_images"
    base_path_2 = r"C:\Users\Mayank\Desktop\augumentation_data\new_data_2"
    obj_img_path = os.path.join(base_path,obj,"images")
    obj_label_path = os.path.join(base_path,obj,"labels")
    # os.makedirs(os.path.join(base_path,f"{obj}_new"))
    output_directory_images = os.path.join(base_path_2,f"{obj}_new","images")
    print(output_directory_images)
    print("aaaaaaaaaaaaaaaaaaaaa")
    output_directory_labels = os.path.join(base_path_2,f"{obj}_new","labels")
    print(output_directory_labels)
    count = 0
    
    for obj_single_img_path , obj_single_label_path in zip(os.listdir(obj_img_path),os.listdir(obj_label_path)) :
        with open(os.path.join(obj_label_path, obj_single_label_path),"r") as f :
            count += len(f.readlines())
        if count >= instances//2:
            num+=1
            count = float(-100000000)

        obj_single_img_path = os.path.join(obj_img_path, obj_single_img_path)
        obj_single_label_path = os.path.join(obj_label_path, obj_single_label_path)
        augument_yolo_data(obj_single_img_path,obj_single_label_path,output_directory_images,output_directory_labels,img_name_suffix="_aug",num_aug=num)

        