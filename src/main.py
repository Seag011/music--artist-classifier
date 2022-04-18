# Copyright (c) 2021, Pavel Alexeev, pavlik3312@gmail.com
# All rights reserved.
#
# This source code is licensed under the CC BY-NC-SA 4.0 license found in the
# LICENSE file in the root directory of this source tree.



#import .utility as utility
import model as models
import preprocessor as process_files

import random

import tensorflow as tf
import pathlib

import os
import ast
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from keras.layers import Input, Dense
from keras.models import Sequential, load_model

from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam

from work import Work
from train import Train
from preprocessorManeger import PreprocessorManager



def start(isTrain, dataset_processed=False,          
            learn_artist_dir=None,           
            spectrogram_out_dir='spectrogramm',
            work_artist_dir = "work",
            save_metrics_folder='metrics',
            save_weights_folder='weights',
            dataset_name = "default_name",

            sample_rate = 16000,
            channels = 1,
            slice_length = 30000,            
            n_fft = 512,
            hop_length = 512,               
            n_mel = 128, 
            layering = 2,  
                    
            load_checkpoint=False,
            save_metrics=True,
            
            batch_size = 8,
            nb_epochs  = 10,
            early_stop = 2,
            learning_rate = 0.0001,
            random_state = 42):
    """ 
    Parameters
    ------
        train : bool
            start NN as train or work for classification.

        learn_artist_dir : str     
            path for classified audio files.

        spectrogram_out_dir : str
            path for spectrograms for learning NN.

        work_artist_dir : str
            path for files that will be classified.

        save_metrics_folder : str
            path for metric of current model.

        save_weights_folder : str
            path for saving current trained NN weights.

        slice_length : int
            length of parts to be classified over audio track.

        layering : int 
            [0 .. 10], overlapping parameter of parts of track.        
    """ 
    print(work_artist_dir)
    work_artist_dir = os.path.normpath(work_artist_dir)
    sep = str("_")
    ## name for saving metrics and weights
    ## WARNING! Added sep after dataset name
    saveName = dataset_name + sep + str(slice_length) + sep \
                + str(n_fft) + sep \
                + str(hop_length) + sep \
                + str(layering) + sep \
                + str(batch_size) + sep \
                + str(nb_epochs) + sep \
                + str(learning_rate)
    print(saveName)


    index_labels = 0
    if (isTrain):
        if (not dataset_processed):
            p = PreprocessorManager(learn_artist_dir, 
                                    spectrogram_out_dir, 
                                    fragment_size_ms = slice_length, 
                                    sample_rate = sample_rate,
                                    n_fft = n_fft,
                                    hop_length = hop_length,
                                    layering = layering)
            p.process()
        trainer = Train(random_state)
        index_labels = trainer.startTrain(spectrogram_out_dir,                            
                                          save_metrics_folder,
                                          save_weights_folder,
 
                                          saveName,
 
                                          batch_size,
                                          nb_epochs,
                                          early_stop,
                                          learning_rate)
        ## save labels to file
        with open(save_weights_folder+os.sep+"labels_"+saveName, 'w') as f: 
            f.write(repr(index_labels))

    #index_labels = {'_NONE_': 0, 'Белый Игорь': 1, 'Городецкий Олег': 2, 'Иванов Игорь (Томск)': 3, 'Труханов Сергей': 4, 'Юлий Ким': 5}
    with open(save_weights_folder+os.sep+"labels_"+saveName, 'r') as f: 
        index_labels = ast.literal_eval(f.read())
    print(index_labels)
    worker = Work()
    worker.startWork(work_artist_dir,                                    
                     save_weights_folder,
                     save_metrics_folder,                  
                     saveName,
     
                     sample_rate,
                     channels, 
                     slice_length,
                     n_fft, 
                     hop_length,
                     n_mel, 
                     layering,
 
                     index_labels)

# start(False, dataset_processed=True, batch_size=4, nb_epochs=1, learning_rate=0.001,
#         slice_length = 50000, layering=1,
#         learn_artist_dir=str(r"I:\Downloads\Треки для Павла Алексеева"),
#         work_artist_dir=str(r"I:\Downloads\spectrograms\work_dir\\"),
#         spectrogram_out_dir = str(r"I:\Downloads\spectrograms\Треки для Павла Алексеева(50000ms)"),
#         save_weights_folder=str(r"J:\Jupyter\Jupyter\weights"),
#         save_metrics_folder=str(r"J:\Jupyter\Jupyter\metrics"),        
#         dataset_name = "Bards")


# start(isTrain=False, dataset_processed=True, batch_size=4, nb_epochs=25, learning_rate=0.0003,
#         slice_length = 10000, layering=2,
#         learn_artist_dir=str(r"I:\Downloads\artist20-mp3s-32k\artist20\mp3s-32k"),
#         work_artist_dir=str(r"I:\Downloads\artist20-mp3s-32k\artist20\work"),
#         spectrogram_out_dir = str(r"I:\Downloads\artist20-mp3s-32k\artist20\spectrograms"),
#         save_weights_folder=str(r"I:\Downloads\artist20-mp3s-32k\artist20\weight"),
#         save_metrics_folder=str(r"I:\Downloads\artist20-mp3s-32k\artist20\metrics"),        
#         dataset_name = "artist20")
