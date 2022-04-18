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
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from keras.layers import Input, Dense
from keras.models import Sequential, load_model

from keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam 


import datetime
import shutil

### utils ###

def getImageShape(path):
    return tf.image.decode_png(tf.io.read_file(path)).shape

#############

def rmdirIsExist(path):
    if (os.path.exists(path)):
        shutil.rmtree(path)

def choiceStrategyMaxSum(predictions):
    sum = np.zeros(len(predictions[0]))
    for elem in predictions:
        sum = np.add(sum, elem)    

    return sum/len(predictions)

def choiceStrategyVoteMax(predictions):
    '''
        Returns np.array with predicted indexes for each track part 
        index is chosen as index of max of all values of prediction possibilities
    '''

    result = np.zeros(len(predictions[0]))
    result = np.argmax(predictions, axis=1)
    print(result, flush=True)   
    result = np.bincount(result)    
    return result

class Work:

    fileOutPath = ""
    out = "" # out file

    def getDirByFileName(self, file):
        filename = os.path.basename(file)
        cur_dir = os.path.dirname(os.path.abspath(file))

        return cur_dir + os.sep + "temp" + os.sep + filename #+str("_spec")

    def prepareFolders(self, work_artist_dir):
        # get filenames
        files = os.listdir(work_artist_dir)
        files = [work_artist_dir + os.sep + file for file in files]
        # filter files from directories
        files = [file for file in files if os.path.isfile(file)]
        
        #print (files)
        rmdirIsExist(work_artist_dir + os.sep + "temp" + os.sep)

        # create directories named by filenames         
        for elem in files:        
            if (os.path.isfile(elem)):
                folder = self.getDirByFileName(elem)
                if (not os.path.exists(folder)):                
                    os.makedirs(folder)
                # filename = os.path.basename(elem)
                # cur_dir = os.path.dirname(os.path.abspath(elem))
                # os.mkdirs(cur_dir+os.path+"temp"+os.sep+filename+str("_spec"))

        # get created directories
        temp_dir = work_artist_dir + os.sep + "temp" + os.sep

        dirs = os.listdir(temp_dir)     
     
        dirs = [temp_dir+dir for dir in dirs]
        print (dirs)
        return dirs

    def loadModel(self, save_weights_folder, save_model_name):        
        model = load_model(save_weights_folder+os.sep+save_model_name+os.sep+
                           save_model_name+".h5")
        model.summary()
        return model


    def process(self, dirs, work_artist_dir, index_labels, model, sample_rate,
                       channels, 
                       fragment_size_ms,
                       n_fft,
                       hop_length,
                       n_mel,
                       layering):

        for dir in dirs:            
            # remove spec to find file
            file = dir.replace("_spec", "")
            file = dir.replace("temp"+os.sep, "")
            #print(file, flush=True)
            ## process file and create all spectrogramms
            #uniqueName = str(dir).split(os.path)[-1]
            #uniqueName = "".join(uniqueName)
            #print(fragment_size_ms, flush=True)
            f = process_files.Preprocessor(file, dir, 222, 
                        sample_rate=sample_rate,
                        channels=channels,
                        fragment_size_ms=fragment_size_ms,
                        n_fft=n_fft,                        
                        hop_length=hop_length,
                        n_mel=n_mel,
                        layering=layering)
            f.saveToDir(None)
            # get resulted spectrograms
            spectrograms = os.listdir(dir)
            spectrograms = [dir+os.sep+elem for elem in spectrograms]
            
            # print(spectrograms, flush=True)
            # create dataset from spectrograms
            data = tf.data.Dataset.from_tensor_slices(spectrograms)
            #print(data)
            
            INPUT_SHAPE_IMG = getImageShape(spectrograms[0])
            #print(INPUT_SHAPE_IMG)
            INPUT_SHAPE_IMG = (INPUT_SHAPE_IMG[0], INPUT_SHAPE_IMG[1])
            ##model = getModel(INPUT_SHAPE_IMG, len(index_labels), learning_rate)

            ##model.load_weights(save_weights_folder + os.sep + saveName)

            def preprocess_image(path):
                image = tf.io.read_file(path)
                image = tf.image.decode_png(image, channels=3)
                image = tf.image.resize(image, INPUT_SHAPE_IMG)
                print(image.shape, flush=True)
                # normalize to [0,1] range
                image = tf.cast(image, tf.float32) / 255.0  
                return image

            # use dataset as images
            data = data.map(preprocess_image)
            data = data.batch(1)
            
            # predict and get final result
            result = model.predict(data, batch_size = 1)

            print(result.shape, flush=True)

            #result = choiceStrategyMaxSum(result)
            choice = choiceStrategyVoteMax(result)

            #print(result)
            
            index = np.argmax(choice)
            # def getKeysByValue(dictOfElements, valueToFind):
            #     listOfKeys = list()
            #     listOfItems = dictOfElements.items()
            #     for item  in listOfItems:
            #         if item[1] == valueToFind:
            #             listOfKeys.append(item[0])
            #     return  listOfKeys

            # key = getKeysByValue(index_labels, index)
            key = index_labels[index]
            print(file, "   ", choice, flush=True)
            print(index, "   ", key, flush=True)
            print(datetime.datetime.now(), flush=True)
            ## output to file            
            self.out.write(file+"\t\t"+str(key)+"\t\t"+str(choice[index])+"/" 
                       + str(result.shape[0])+" "+str(choice[index]/result.shape[0])+"\n")  
        
   
    def startWork(self,
                  work_artist_dir,                  
                  save_weights_folder,
                  save_metrics_folder,
                  save_model_name,
                  save_results_name,
                  sample_rate, 
                  channels, 
                  fragment_size_ms,
                  n_fft, 
                  hop_length,
                  n_mel, 
                  layering,            
                  index_labels):

        
        dirs = self.prepareFolders(work_artist_dir)
        print(dirs, flush=True)
        self.fileOutPath = save_results_name + str(".txt")
        self.out = open(self.fileOutPath, 'w', encoding="utf-8")
        print(self.fileOutPath, flush=True)
        model = self.loadModel(save_weights_folder, save_model_name)
        self.process(dirs, work_artist_dir, index_labels, model, sample_rate, channels, 
                     fragment_size_ms, n_fft, hop_length, n_mel, layering)
        self.out.close()