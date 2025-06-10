


# with open("object_names.txt",'r') as f:
#         objs = f.read().strip().split(' ')

# def load_data(images_path, labels_path):
#     #makes an array of all image files inna particular directory
#     image_files = [f for f in os.listdir(images_path) if f.endswith(('.jpg', '.png','.jpeg'))]
#     # label_files_unordered = [f for f in os.listdir(labels_path) if f.endswith('.txt')]
    
#     label_files = []

#     for image_file in image_files:

#         #splits @ extension

#         label_file = os.path.join(labels_path,image_file.replace(os.path.splitext(image_file)[1], '.txt'))
#         if os.path.exists(label_file):
#             label_files.append(label_file)
#         else:
#             print(label_file)
#             print('what image is this bro?')
#             pass
    
#     images = []
#     labels = []

#     for image_file, label_file in zip(image_files, label_files):
#         image_path = os.path.join(images_path, image_file)
        
        

#         label_path = label_file
#         with open(label_path, 'r') as f:
#             lines = f.readlines()
            
#         boxes = []

#         count_item = {}

#         for line in lines:
#             parts = line.strip().split()
#             class_index = int(parts[0])  
#             x_center = float(parts[1])
#             y_center = float(parts[2])
#             width = float(parts[3])
#             height = float(parts[4])
#             count_item[class_index] = count_item.get(class_index, 0) + 1
#             boxes.append({
#                 'class': class_index,
#                 'x_center': x_center,
#                 'y_center': y_center,
#                 'width': width,
#                 'height': height
#             })
#         max_count = max(count_item.values())

#         for obj ,count in count_item.items():
#             if count == max_count :
#                 obj_labels = [ box for box in boxes if box['class']== obj ]
#                 base_obj_pth = base_path+objs[obj]+"/labels"
#                 num_files = len([f for f in os.listdir(base_obj_pth) if os.path.isfile(os.path.join(base_obj_pth, f))])
#                 total_obj_pth = base_obj_pth+f"/{num_files}"+".txt"
#                 os.makedirs(total_obj_pth)
#                 with open(total_obj_pth,'w') as f:
#                     for row in obj_labels:
#                         f.write(" ".join(map(str, row)) + "\n")
#                 base_obj_pth2 = base_path+objs[obj]+"/images"
#                 num_files2 = len([f for f in os.listdir(base_obj_pth2) if os.path.isfile(os.path.join(base_obj_pth2, f))])
#                 total_img_pth = base_obj_pth2+f"/{num_files2}"+os.path.splitext(image_path)[1]
                
#                 shutil.move(image_path,total_img_pth)

