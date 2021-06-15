import os
import argparse

parser = argparse.ArgumentParser(description='Parameters for create list')
parser.add_argument('-file_name', help='Set the file name', default='train', type=str)
parser.add_argument('-list_name', help='Set the list name', default='trainlist', type=str)
args = parser.parse_args()

file_name = args.file_name
list_name = args.list_name

file_path = "./data/" + file_name
path_list = os.listdir(file_path+"/hazy")
path_name = []
for i in path_list:
    path_name.append(i)
path_name.sort()
for image_name in path_name:
    with open(file_path+"/" + list_name +  ".txt", "a") as file:
        file.write(image_name + "\n")
    file.close()
print("write finished!")
