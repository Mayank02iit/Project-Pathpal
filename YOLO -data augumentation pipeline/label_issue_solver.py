import os
count = 0
for index,folder in enumerate(os.listdir("new_data_2")):
    for file in os.listdir(f"new_data_2/{folder}/labels"):     
        total_data = []   
        with open(f"new_data_2/{folder}/labels/{file}","r") as f:
            total_data = f.readlines()            # Total_data = f.readlines()
            for i , data in enumerate(total_data):
                new_data = data.strip().split(" ")
                num = new_data[0].split(".")[0]
                new_data[0]=str(index)
                total_data[i]=" ".join(new_data)
                # f.write(" ".join(new_data))
        # f"new_data_2/{folder}/labels/{file}"
        with open(f"new_data_2/{folder}/labels/{file}","w") as f:
            for line in total_data:
                f.write(line)
                f.write("\n")
# count = 0

# for folder in os.listdir("new_data_2"):
#     for file in os.listdir(f"new_data_2/{folder}/labels"):  
#         count+=1   
#         total_data = []   
#         new_data = []
#         with open(f"new_data_2/{folder}/labels/{file}","r") as f:
#             total_data = f.read().split(" ")
#             new_data.append(" ".join(total_data[0:4]))
#             # print(new_data)
#             for i in range(4,len(total_data)-3,3):
#                 array = [total_data[0]] + total_data[i:i+3]
#                 new_data.append(" ".join(array))
#             # print(new_data)    
#                      # Total_data = f.readlines()
#         #     for i , data in enumerate(total_data):
#         #         new_data = data.split(" ")
#         #         num = new_data[0].split(".")[0]
#         #         new_data[0]=str(num)
#         #         total_data[i]=" ".join(new_data)
#         #         # f.write(" ".join(new_data))
#         with open(f"new_data_2/{folder}/labels/{file}","w") as f:
#             for line in new_data:
#                 f.write(line)
#                 f.write("\n")



                
            