import os
objs = ['Apple', 'Banana', 'Bell Pepper', 'Carrot', 'Detergent', 'Egg', 'Lemon', 'Orange', 'Strawberry']

with open("object_counts.txt","r") as f:
    data = f.readlines()
object_dict = {}
# for line in data:
#     a = line.strip().split(" ")
#     c = line.strip().split('\\')[6]
#     object_dict[c][int(a[-1])] = object_dict.get(c, 0) + int(a[-1])

for line in data:
    a = line.strip().split(" ")
    c = line.strip().split("\\")[6]  # Extract object category
    
    if c not in object_dict:
        object_dict[c] = {}  # Initialize a nested dictionary

    key = int(a[-1])  # Assuming the second-last value is the key
    value = int(a[-1])  # Assuming the last value is the count
    
    object_dict[c][key] = object_dict[c].get(key, 0) + 1


print(object_dict)
count_tracker_aug= { }
count_tracker_aug['Bread']=2306
count_tracker_aug['Drinks']=2397
import os
for item in objs:
# item = "Apple"
    directory = rf"C:\Users\Mayank\Desktop\augumentation_data\{item}_new\labels"
    
    count = sum(len(open(os.path.join(directory, file), 'r').readlines()) for file in os.listdir(directory) if os.path.isfile(os.path.join(directory, file)))
    count_tracker_aug[item]= count_tracker_aug.get(item,0)+count

print(count_tracker_aug)