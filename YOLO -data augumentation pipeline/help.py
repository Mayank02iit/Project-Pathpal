import os
count = 0
base_path = r"C:\Users\Mayank\Desktop\augumentation_data\new_data_2\Apple_new\labels"
for f in os.listdir(base_path):
    with open(os.path.join(base_path,f),"r") as file:
        count+=len(file.readlines())
print(count)
