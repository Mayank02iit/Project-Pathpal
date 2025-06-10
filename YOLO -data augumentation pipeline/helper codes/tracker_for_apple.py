import os 
# path = r"C:\Users\Mayank\Desktop\augumentation_data\train\labels"
# count = 0
# count_dict = {}
# for file in os.listdir(path):
#     full_path = os.path.join(path,file)
#     with open(full_path,"r") as f:
#         count+= len(f.readlines())
    
# print(count)
# no_aug_images
# C:\Users\Mayank\Desktop\augumentation_data\no_aug_images\Banana\labels
count_dict = {}
items =['Apple', 'Banana', 'Bell Pepper', 'Bread', 'Carrot', 'Detergent', 'Drinks', 'Egg', 'Lemon', 'Orange', 'Strawberry']
for item in items:
    path = fr"C:\Users\Mayank\Desktop\augumentation_data\new_data_2\{item}_new\labels"

    count = 0
    for file in os.listdir(path):
        full_path = os.path.join(path,file)
        with open(full_path,"r") as f:
            count += len(f.readlines())
    count_dict[item]=count
print(count_dict)