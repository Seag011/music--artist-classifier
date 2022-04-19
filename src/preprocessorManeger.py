# Copyright (c) 2021, Pavel Alexeev, pavlik3312@gmail.com
# All rights reserved.
#
# This source code is licensed under the CC BY-NC-SA 4.0 license found in the
# LICENSE file in the root directory of this source tree.



import os
import pathlib
import random
import sys
import numpy as np



from preprocessor import Preprocessor

import multiprocessing as mp
 
class PreprocessorManager:

    __data_path = ""
    __spectr_out_path = ""

    # ## DEPRECATED:
    # ## TODO: Refactor code with this variable
    # __mode = 0 #DEPRECATED
    __classes = []   
    __audio_extentions = [ 
            "mp3", "wav", "flac", "aac", "ogg", "wma", "m4a"
        ]

    sample_rate = 16000
    channels = 1
    fragment_size = 1000
    n_fft = 512
    layering = 1
    hop_length = 512


    def __init__(self, 
                    data_path,
                    spectr_out_path, 
                    audio_extentions,
                    fragment_size_ms,                    
                    sample_rate = 16000,
                    channels = 1,
                    n_fft = 512,                    
                    hop_length = 512,
                    layering = 1):

        #normalize paths
        self.__data_path        = os.path.normpath(data_path)
        self.__spectr_out_path  = os.path.normpath(spectr_out_path)
        self.fragment_size      = fragment_size_ms
        self.sample_rate        = sample_rate
        self.channels           = channels
        self.n_fft              = n_fft
        self.layering           = layering
        self.hop_length         = hop_length
        self.__audio_extentions = audio_extentions

    def is_audio(self, path):
        """Check allowed audio formats"""

        fileName = os.path.split(path)[-1]        
        extention = fileName.split('.')[-1]
        extention = extention.lower()
        
        return extention in self.__audio_extentions

    # def __getClasses(self): 
    #     self.__classes = ["Non-of-the-list"] + os.listdir(self.__data_path)
    #     return self.__classes
    
    def __getClass(self, path):
        """Get classes from data path"""
        path = path.replace(self.__data_path + os.sep,"")

        #print(path)
        pased_path = path.split(os.path.sep)

        return pased_path[0]

    # def getOutputArrayClass(self, pos, size):
    #     """DEPRECATED"""
    #     #print("class len", len(self.__classes))
        
    #     arr = np.zeros((size, int(len(self.__classes))))
    #     for elem in arr:
    #         elem[pos] = 1.0       
    #     print(np.zeros(4).shape[0])
    #     return arr

    def __getListOfFiles(self, path):
        """ 
            Create a list of file and sub directories              
        """

        listOfFile = os.listdir(path)
        allFiles = list()
        # Iterate over all the entries
        for entry in listOfFile:
            # Create full path
            fullPath = os.path.join(path, entry)
            # If entry is a directory then get the list of files in this directory 
            if os.path.isdir(fullPath):
                allFiles = allFiles + self.__getListOfFiles(fullPath)
            else:
                if (self.is_audio(fullPath)):
                    allFiles.append(fullPath)
                
        return allFiles

    def checkPaths(self):
        """ 
            Check wether `paths` are dirs and if it exists\n
            Create `out_dir` if its not exists
        """
        if(os.path.isdir(self.__data_path) and os.path.exists(self.__data_path)):
            #if(os.path.isdir(self.__spectr_out_path)):
                if (not os.path.exists(self.__spectr_out_path)):
                    os.makedirs(self.__spectr_out_path)

                
                return True
        return False

    # def __getNameWithClass(self, path):
    #     """ DEPRECATED

    #     RETURNS
    #     -------
    #         if mod == 1 name format is class_album_uniqeIDSong_
    #         if mod == 0 name format is class__uniqeIDSong_            
    #     """
    #     sep = "_"        
    #     uniqueIdSong = str(random.randint(0, sys.maxsize))
        
    #     if (self.__mode):
            
    #         name = path[-3] + sep + path[-2] + sep + uniqueIdSong + sep
    #     else:
    #         name = path[-2] + sep + uniqueIdSong + sep
    #     return name.replace(" ", "_")

    def __getClasses(self): 
        self.__classes = os.listdir(self.__data_path)
        return self.__classes

    def createDirs(self):        
        self.__getClasses()
        for dir in self.__classes:
            path = self.__spectr_out_path + os.sep + dir            
            if (not os.path.exists(path)):                
                os.makedirs(path)

    def process(self):      
        """Convert all audio files to spectrograms with specified parameters

        Returns
        -------
        `False` if there occupied an error or `none` if operation successfully  
        completed
            
        """
        
        print(self.__data_path, flush=True)
        print(self.__spectr_out_path, flush=True)

        if (self.checkPaths()):
            
            print("Started conversion to spectrograms...", flush=True)
            ## getting all files in subfolders
            files = self.__getListOfFiles(self.__data_path)
            print("In derectories found: ", len(files), " files", flush=True)
            print("Starting convertion to spectrograms...", flush=True)

            self.createDirs()

            for file in files:
                
                ## generating unique name by mode
                uniqueName = self.__getClass(file)
                ## process file and create all spectrogramms
                f = Preprocessor(file, self.__spectr_out_path, uniqueName, 
                            sample_rate=self.sample_rate,
                            channels=self.channels,
                            fragment_size_ms=self.fragment_size,
                            n_fft=self.n_fft,
                            layering=self.layering)
                f.saveToDir(uniqueName)

            print("Convertion over.", flush=True)
        else:
            print("One of the paths is wrong. Check paths and try over.", flush=True)
            return False
    
    # def processArrays(self):
    #     """DEPRECATED"""
    #     #print(self.__data_path)
    #     #print(self.__spectr_out_path)
    #     if (self.checkPaths()):
            
    #         print("Started conversion to spectrograms...")
    #         ## getting all files in subfolders
            
    #         files = self.__getListOfFiles(self.__data_path)

    #         self.__classeses = self.__getClasses()
    #         print(self.__classeses)

    #         fragments = False
    #         classification = False
    #         is_first = True
    #         for file in files:
    #             ## generating name by mode
    #             name = self.__getNameWithClass(file)
                
    #             ## getting class id of current file
    #             audio_class = self.__getClass(file)               
    #             ## process file and create all spectrogramms
    #             f = Preprocessor(file, self.__spectr_out_path, name)                
    #             S = f.getSpectrogramsArray()
                
    #             current_classes = self.getOutputArrayClass(audio_class, len(S))
    #             print(len(current_classes))
    #             #print()
    #             if (is_first):                   
    #                 fragments = S
    #                 classification = current_classes
    #                 is_first = False
    #             else:
    #                 fragments = np.concatenate((fragments, S))
    #                 classification = np.concatenate((classification, current_classes)) 
    #         #print(type(result[0]))
    #         #print(len(result))
    #         print("Convertion over.")
    #         #print(classification)
    #         return fragments, classification
    #     else:
    #         print("One of the paths is wrong. Check paths and try over.")
    #         return False

# data_path = r"C:\Users\Developer\Desktop\AudioNN\Jupyter\data"
# out_path  = r"C:\Users\Developer\Desktop\AudioNN\Jupyter\specs"
# data_path = r"I:\Downloads\Треки для Павла Алексеева"
# out_path  = r"I:\Downloads\spectrograms\Треки для Павла Алексеева(5000ms)"
# p = Transfer(data_path, out_path, mode = 0, fragment_size_ms=5000)
# p.process()