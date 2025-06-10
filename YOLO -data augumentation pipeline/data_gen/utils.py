import os
import yaml

with open(r'C:\Users\Mayank\Desktop\augumentation_data\dataset\data.yaml', 'r') as file:
    data = yaml.safe_load(file)  # Safely load YAML into a Python dict[1][7]

names = data['names']  # Directly access the list[6][9]
for name in names:
    if os.path.exists(name):
        os.remove(name)
    os.mkdir(name)
    os.mkdir(name+"/images")
    os.mkdir(name+"/labels")
    

with open('object_names.txt','w') as f:
    for name in names:
        f.write(name+" ")
# with open("object_names.txt",'r') as f:
#         objs = f.read().strip().split(" ")

# print(objs)