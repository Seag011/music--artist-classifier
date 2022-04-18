# Copyright (c) 2021, Pavel Alexeev, pavlik3312@gmail.com
# All rights reserved.
#
# This source code is licensed under the CC BY-NC-SA 4.0 license found in the
# LICENSE file in the root directory of this source tree.


import os
 
# def get_list_of_files_(self, path):
#     # create a list of file and sub directories 
#     # names in the given directory 
#     listOfFile = os.listdir(path)
#     allFiles = list()
#     # Iterate over all the entries
#     for entry in listOfFile:
#         # Create full path
#         fullPath = os.path.join(path, entry)
#         # If entry is a directory then get the list of files in this directory 
#         if os.path.isdir(fullPath):
#             allFiles = allFiles + self.__getListOfFiles(fullPath)
#         else:
#             if (self.is_audio(fullPath)):
#                 allFiles.append(fullPath)
            
#     return allFiles

def list_to_file(path, list):
    with open(path, "w") as output:
        for elem in list:
            output.write(str(elem)+"\n")

def get_list_of_files(path):
    # create a list of file and sub directories
    tree = os.walk(path)
    allFiles = list()
    for root, dirs, files in tree:
        if (not files): # files list
            for elem in files:
                allFiles.append(path + os.sep + elem)
        if (not dirs): # dirs list
            for dir in dirs:
                allFiles += get_list_of_files(os.path.join(root, dir))
    return allFiles