# Copyright (c) 2021, Pavel Alexeev, pavlik3312@gmail.com
# All rights reserved.
#
# This source code is licensed under the CC BY-NC-SA 4.0 license found in the
# LICENSE file in the root directory of this source tree.





#import .utility as utility
# import model as models
# import preprocessor as process_files

# import random

# import tensorflow as tf
# import pathlib


# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns

# from keras.layers import Input, Dense
# from keras.models import Sequential, load_model

# from keras.callbacks import ModelCheckpoint, EarlyStopping
# from keras.optimizers import Adam
from logging import ERROR
import os
import getopt
import ast
#from re import split
import sys
from time import time_ns

from work import Work
from train import Train
from preprocessorManeger import PreprocessorManager

import configuration 

def rmdirIsExist(path):
    if (os.path.exists(path)):
        os.rmdir(path)


class ProgramManager:
    config = ""
    mode = ""

    # constant values #
                      #
    channels    = 1   #      
    n_fft       = 2048#
    hop_length  = 512 #             
    n_mel       = 128 #
    patience    = 10  #
    ###################

    ### configuration setting ###
    dataset_dir        = ""
    temp_files_dir     = ""
    train_results_dir  = ""
    work               = ""

    dataset_name   = ""

    sample_rate    = ""
    slice_length   = ""
    layering       = ""
    filters        = ""
    
    epoches        = ""
    learning_rate  = ""
    batch_size     = ""
    early_stop     = ""
    #############################
    
    ### generating save names ###
    save_name_dataset = ""
    save_name_listfile = ""
    save_name_spectrs = ""
    save_name_model = ""    
    save_name_result = ""
    #############################
    
    ## save directories ##
    save_dir_spectrs = ""
    save_dir_metrics = ""
    save_dir_weights = ""
    save_dir_results = ""    
    ######################

    random_state = "777"

    def __generate_names(self):
        sep = "_"
        self.save_name_model = str(self.sample_rate)+ sep \
                + str(self.slice_length) + sep \
                + str(self.n_fft) + sep \
                + str(self.hop_length) + sep \
                + str(self.layering) + sep \
                + str(self.batch_size) + sep \
                + str(self.epoches) + sep \
                + str(self.learning_rate)
                ## добавить генерацию имен папок и файлов        
                
## added sample rate 
        self.save_name_spectrs = str(self.sample_rate) + sep \
                                + str(self.slice_length) + sep \
                                + str(self.n_fft) + sep \
                                + str(self.hop_length) + sep \
                                + str(self.layering)
        self.save_name_listfile = "included_files.list"

        os.makedirs(self.work_dir + os.sep + "results", exist_ok=True)
        self.save_name_result = self.work_dir + os.sep + "results" + os.sep + "result_" + self.dataset_name + sep + str(time_ns())
        
    def __generate_dirs(self):
        self.save_dir_spectrs = self.temp_files_dir+os.sep+ \
                                self.dataset_name+os.sep+self.save_name_spectrs

        self.save_dir_metrics = self.train_results_dir+os.sep+"metrics"
        self.save_dir_weights = self.train_results_dir+os.sep+"weights"
        self.save_dir_results = self.work_dir + os.sep + "results" 

        os.makedirs(self.save_dir_metrics, exist_ok=True)
        os.makedirs(self.save_dir_weights, exist_ok=True)
        
    def filter_parse(self):
        #print(self.filters)
        self.filters = self.filters.split(";")

    def __init_config(self):
        self.dataset_dir        = self.config['Directories']['dataset_dir']
        self.temp_files_dir     = self.config['Directories']['temp_files']
        self.train_results_dir  = self.config['Directories']['train_results']
        self.work_dir           = self.config['Directories']['work']

        self.dataset_name   = self.config['DatasetProcess']['dataset_name']
        self.slice_length   = int(self.config['DatasetProcess']['slice_length'])
        self.layering       = int(self.config['DatasetProcess']['layering'])
        self.sample_rate    = int(self.config['DatasetProcess']['sample_rate'])
        self.filters        = str(self.config['DatasetProcess']['filters'])

        self.epoches        = int(self.config['TrainParameters']['epoches'])
        self.learning_rate  = float(self.config['TrainParameters']['learning_rate'].replace(",","."))
        self.batch_size     = int(self.config['TrainParameters']['batch_size'])
        self.early_stop     = bool(self.config['TrainParameters']['early_stop'])
        self.patience       = int(self.config['TrainParameters']['patience'])

        self.n_fft          = int(self.config['Spectrogram']['n_fft'])
        self.hop_length     = int(self.config['Spectrogram']['hop_length'])
        
        # parse filters
        self.filter_parse()

    def __init__(self, config, mode):
        self.config = config
        self.mode = mode

        self.__init_config()
        self.__generate_names()
        self.__generate_dirs()


    def process_dataset(self):
        
        p = PreprocessorManager \
                        (self.dataset_dir, 
                        self.save_dir_spectrs,
                        audio_extentions=self.filters,
                        fragment_size_ms = self.slice_length, 
                        sample_rate = self.sample_rate,
                        n_fft = self.n_fft,
                        hop_length = self.hop_length,
                        layering = self.layering)
        p.process()

    def train(self):

        trainer = Train(self.random_state)
        index_labels = trainer.startTrain(self.save_dir_spectrs,                            
                                          self.save_dir_metrics,
                                          self.save_dir_weights,
 
                                          self.save_name_model,
 
                                          self.batch_size,
                                          self.epoches,
                                          self.early_stop,
                                          self.patience,
                                          self.learning_rate)
        ## save labels to file
        with open(self.save_dir_weights+os.sep+"labels_"+ 
                  self.save_name_model, 'w') as f: 
            f.write(repr(index_labels))

    def work(self):
        index_labels = ""
        with open( self.save_dir_weights + os.sep + "labels_" + 
                   self.save_name_model, 'r') as f: 
            index_labels = ast.literal_eval(f.read())

        print(index_labels, flush=True) 
        
        worker = Work()
        worker.startWork(self.work_dir,     
                        self.save_dir_weights,                               
                        self.save_dir_metrics,                        
                        
                        self.save_name_model,
                        self.save_name_result,
        
                        self.sample_rate,
                        self.channels, 
                        self.slice_length,
                        self.n_fft, 
                        self.hop_length,
                        self.n_mel, 
                        self.layering,
                        
                        index_labels)
                        

    def all(self):
        self.process_dataset()
        self.train()
        self.work()
    
    def prepareWholeDataAndModel(self):
        self.process_dataset()
        self.train()
        

configPath = ''
mode = ''

def checkInput(configPath, mode):
    print("mode: ", configPath, mode in ("prepare", "train", "work", "all"), flush=True)
    print("path: ", mode, os.path.exists(configPath), flush=True)
    return not os.path.exists(configPath) or \
           not (mode in ("prepare", "train", "work", "all"))

def main(argv):
    configPath = ''
    mode = ''
    print
    try:
        opts, arg = getopt.getopt(argv,"hi:m:",["help,config=,mode="])
    except getopt.GetoptError:
        print('Use -h to get help')
        sys.exit(2)
    print(opts)
    for opt, arg in opts: 
        if opt in ("-h", "--help"):
            print("\nA classificator of music tracks by artists.\n" + \
                   "CLASSIFY -i <setting_path.ini> -m <prepare|train|work|all> [-h] \n\n" + \
                   "-i, --config \tpath to setting ini file\n"+ \
                   "-m, --mode \tmode of algorithm:\n\t\t" + \
                        "  prepare - preparing dataset\n\t\t" + \
                        "  train   - learn Neural Network\n\t\t" + \
                        "  work    - start pretrained model on selected data\n\t\t" + \
                        "  all     - start all the previous steps one by one\n",
                    flush=True)
            sys.exit(0)   
        elif opt in ("-i", "--config"):
            configPath = arg
            print (arg)
        elif opt in ("-m", "--mode"):
            mode = arg        
        else:
            print("Config path is not presented!", flush=True)
            sys.exit(2)
    if (checkInput(configPath, mode)):
        print()
        print("Agruments value wrong!", flush=True)
        sys.exit(2)

    print('Input file is "', configPath, flush=True)
    return configPath, mode

if __name__ == "__main__":
   configPath, mode = main(sys.argv[1:])

config = ""
try:
    config = configuration.getConfig(configPath)
except Exception:
    print("Error! Configuration file not found or corrupted!", flush=True)
    sys.exit(2)


manager = ProgramManager(config, mode)
if (mode == "prepare"):
    manager.process_dataset()
elif (mode == "train"):
    manager.train()
elif (mode == "work"):
    manager.work()
elif (mode == "all"):
    manager.process_dataset()
    manager.train()
    manager.work()
    

# example of input C>main_config.py 

# start(isTrain=False, dataset_processed=True, batch_size=4, nb_epochs=25, learning_rate=0.0003,
#         slice_length = 10000, layering=2,
#         learn_artist_dir=str(r"I:\Downloads\artist20-mp3s-32k\artist20\mp3s-32k"),
#         work_artist_dir=str(r"I:\Downloads\artist20-mp3s-32k\artist20\work"),
#         spectrogram_out_dir = str(r"I:\Downloads\artist20-mp3s-32k\artist20\spectrograms"),
#         save_weights_folder=str(r"I:\Downloads\artist20-mp3s-32k\artist20\weight"),
#         save_me   trics_folder=str(r"I:\Downloads\artist20-mp3s-32k\artist20\metrics"),        
#         dataset_name = "artist20")


        ## add to config: 
        #       random_state
        #       isTrained
        #       isProcessed
