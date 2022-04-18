# Copyright (c) 2021, Pavel Alexeev, pavlik3312@gmail.com
# All rights reserved.
#
# This source code is licensed under the CC BY-NC-SA 4.0 license found in the
# LICENSE file in the root directory of this source tree.


import os
import utilits

class FileManager:
    cfg_name = ""

    # define directories names
    work_dir = "work"
    transformed_dataset = "spectrograms"
    weights = "weights"
    metrics = "metrics"
    results = "results"
    out_files = "out_files"

    # define directories path
    dataset_path = ""
    transformed_dataset_path = ""
    out_path = ""

    list_file = "file.list"

    def create_folder_structure(self):
        # creatin whole folders structures
        transformed_name = # THINK ABOUT NAME OF SUB-CFG NAME OF SPECTROGRAMS
        os.makedirs(self.transformed + os.sep + transformed_name + os.sep)
        os.makedirs(self.out_path + os.sep + self.metrics) # ./metrics
        os.makedirs(self.out_path + os.sep + self.weights) # ./weights
        os.makedirs(self.out_path + os.sep + self.results) # ./results

    def FileManager(self, cfg_name, dataset_path, transformed_dataset_path, result_path): 
        self.cfg_name = cfg_name
        self.dataset_path = dataset_path
        self.transformed_dataset_path = transformed_dataset_path
        self.result_path = result_path
    
    ## TODO: add filters to this folder
    def gen_relative_subdirs_paths(self): 
        # generates list of relative files path
        file = self.dataset_path + os.sep + list_file
        current_state = utilits.get_list_of_files(self.dataset_path)

        dif = 0
        if (not (os.path.exists(current_file)):
            utilits.list_to_file(current_state) +.
            # add time and number of files in dataset to file
            
        else            
           dif = self.compare(file, current_state)
        

        
            


    

