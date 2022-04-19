# Copyright (c) 2021, Pavel Alexeev, pavlik3312@gmail.com
# All rights reserved.
#
# This source code is licensed under the CC BY-NC-SA 4.0 license found in the
# LICENSE file in the root directory of this source tree.



import os
import sys
import pathlib

#from matplotlib import pyplot as plt
import numpy as np
from matplotlib import pyplot as plt
from pydub import AudioSegment
import librosa
import librosa.display
#import png
import random
import pylab
import time

import plotly.express as px


import pdb

class Preprocessor:

    __path = ""
    __out_spectr_dir = ""
    __out_spectr_file = ""

    __sample_rate = 16000
    __channels = 1
    __fragment_size_ms = 1000
    __n_fft = 512
    __n_mel = 128
    __layering = 1
    __hop_length = 512

    __PCM = []
    __num_per_fragment = 0
    __fragments = []

    __fragments_number = 0
    __fragment_length = 0

    def __init__(self, audio_path,
                       out_spectr_dir,
                       out_spectr_file,
                       sample_rate = 16000,
                       channels = 1, 
                       fragment_size_ms = 10000,
                       n_fft = 512,
                       hop_length = 512,
                       n_mel = 128,
                       layering = 2):

        self.__path = audio_path
        self.__out_spectr_dir = out_spectr_dir
        self.__out_spectr_file = out_spectr_file
        self.__sample_rate = sample_rate
        self.__channels = channels
        self.__fragment_size_ms = fragment_size_ms
        self.__n_fft = n_fft
        self.__hop_length = hop_length
        self.__n_mel = n_mel
        self.__layering = layering

    def __getNumpyedAudio(self):
        """Open audio file as PCM with pydub and ffmpeg """

        sound = AudioSegment.from_file(self.__path)
        # set channels to mono
        sound = sound.set_channels(self.__channels)
        # set wanted sample rate
        sound = sound.set_frame_rate(self.__sample_rate)

        # getting raw samples PCM of mono audio 
        samples = sound.get_array_of_samples()

        # interpret list as numpy array of np.float32
        # its need to make spectrogram
        fp_arr = np.array(samples).T.astype(np.float32)
        # different formats provides different sample types
        # geting max of this type and finally convert it to float 32
        MAX_ARR_TYPE = np.float32(np.iinfo(samples.typecode).max)
        fp_arr /= MAX_ARR_TYPE

        self.__PCM = fp_arr

    # ## WARNING: deprecated
    # def __getPCMFragments(self):
    #     #DEPRECATED
    #     """
    #     Split PCM to small fragments

    #     layerin : int
    #         layering shows how much time one sample will added to different fragments
    #         max of layering is 10
    #     """

    #     if (self.__layering < 0 or self.__layering > 10):
    #         self.__layering = 1
        
    #     size_to_cut = int(self.__sample_rate * self.__fragment_size_ms / 1000)
    #     self.__fragment_length = size_to_cut
    #     # shift to get layering effects
    #     shift = int(size_to_cut / self.__layering)
    #     ## cutting fragments using mp.split and cmbine all together after all

    #     result = np.array([], dtype=np.float32)

    #     ## number of framnets in whole PCM
    #     self.__fragments_number = int(len(self.__PCM)/ size_to_cut) - 1

    #     ## splitting into fragments by shift default PCM
    #     for cur_shift in range(0, size_to_cut, shift):

    #         ## last fitted element of PCM without shift (starting of 0)
    #         last_elem = self.__fragments_number * size_to_cut

    #         PCM_temp = self.__PCM[cur_shift: last_elem + cur_shift]
    #         ## spit into fragments
    #         splited = np.split(PCM_temp, int(self.__fragments_number))

    #         if (cur_shift == 0):
    #             result = splited
    #         else:   
    #             result = np.concatenate([result, splited])
    #     self.__fragments = result

    
    def __getMelSpectrogram(self, PCM_part):
        """ get mel spectrogram from numpy array """
        s = librosa.feature.melspectrogram(y=PCM_part, sr=self.__sample_rate, 
                                            n_fft=self.__n_fft,
                                            hop_length=self.__hop_length)
        return librosa.power_to_db(s, ref=np.max)

    def __saveMelSpectrogram(self, spectrogram, path):
        """
        Save spectrogram as PNG file using specific filename and path
        """
        # it will be used in further learning
        pylab.figure(figsize=(spectrogram.shape[1], spectrogram.shape[0]), dpi=1)
        pylab.axis('off') # no axis
        axes = pylab.gca()
        axes.set_ylim([-80.0,0])       

        #pylab.figure()
        pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[]) # Remove the white edge    

        #librosa.display.specshow(librosa.power_to_db(spectrogram, ref=np.max))
        librosa.display.specshow(spectrogram)
        
        pylab.savefig(path+".png", bbox_inches=None, pad_inches=0, format="png")
        #pylab.show()
        
        pylab.clf()
        pylab.close()

    # def __saveMelSpectrogram__experiment(self, spectrogram, path):
    #     # save spectrogram as PNG file
    #     # using specific filename and path
    #     # it will be used in further learning

    #     fig = px.density_heatmap(data_frame=spectrogram)
    #     fig.update_traces(showscale=False)
    #     fig.write_image(path+".png")


    ## assosiate audio class with each frgment
    ## WARNING: deprecated
    # def getClassifiedFragment(self, splitedSpectrogram, audio_class):
    #     """
    #     returns: pair - list of spectr data and list of classes 
    #     correspondents to each element of fragment list
    #     """
    #     #return 
    #     return list(map(lambda elem: [elem, audio_class], splitedSpectrogram))
         
    # def splitSpectrogram(self, s, num_of_, axis = 0, pad_value = int(0)):
    #     """
    #     Return list of spectrogram of length
    #     if spectrogram doesnt split without redundancy 
    #     last array pad with pad_value   
    #     S - 2D array
    #     """
    #     #length along time axis
    #     l = s.shape[axis]
    #     spec_shape = s.shape

    #     #print("_____", int(l/length))
    #     end_of_seq = int(l / n) * n
    #     padding_right = 
    #     subarray = np.split(s[:end_of_seq], n)
    #     print(len(subarray))
    #     # print(len(subarray))
    #     # for elem in subarray:
    #     #     print(len(elem))
    #     padded = []
    #     if (end_of_seq != l):
    #         leftower = s[end_of_seq:]
    #         #print(leftower)
    #         #pad with arrays not const

    #         padded = 
    #         padded = [np.pad(leftower, (0, n - len(leftower)), constant_values=(0,pad_value))]
    #         print(padded[0].shape)
    #         subarray = np.concatenate((subarray, padded)) 
    #     print(len(subarray))
    #     return subarray

    def splitSpectrogram(self, s, len_of_fragment : int, axis = 0, pad_value = -80.0):
        """
        Returns
            list of spectrogram of `length`
            if spectrogram doesnt split without `redundancy`
            last array pad with pad_value   
            S - 2D array
        """
        #print(len(s), " ", len_of_fragment)
        time = s.shape[axis]
        freq = s.shape[1 - axis]
        
        ## reminder of last fragment in audio
        reminder = time % len_of_fragment
        
        ## if part of spectrogramm is more then half empty
        ## then delete it
        if  (reminder >= 0.5 * len_of_fragment):        
            #concate to the end to get optimal division

            #create padding with pad_value
            pad_num = len_of_fragment - reminder  
            pad = np.full((pad_num, freq), pad_value)
            s = np.concatenate((s, pad))
            #print("add empty")
        elif (reminder != 0):            
            s = s[:-reminder]
            #print("remove")

        n = s.shape[axis] / len_of_fragment
        #print(len(s), " ", n)
        #
        subarray = np.split(s, n)

        return subarray
        
    def getSpectrogramsArray(self):
        """
        Returns
            array of splitted spectrogramms
        """
        self.__getNumpyedAudio()
        spectrogramm = np.transpose(self.__getMelSpectrogram(self.__PCM)) 

        if (self.__layering < 0 or self.__layering > 10):
            self.__layering = 1
        
        ## number of spectrogram lenght per one fragment
        ## spectrogram has a different length despite of PCM
        self.__num_per_fragment = int(np.ceil(self.__sample_rate / np.float32(self.__hop_length)))

        ## length of spectrogram fragmet
        ## that could fit in fragment setted by milliseconds
        frag_length = int(self.__num_per_fragment * self.__fragment_size_ms / 1000.0)

        ## 
        delta = int(frag_length / np.float(self.__layering))
        #print("delta = ", delta)

        ## splitting into fragments by shift default PCM
        result = []
        
        for cur_shift in range(0, frag_length, delta):            
            splited = self.splitSpectrogram(spectrogramm[cur_shift:], frag_length, pad_value=-80.0)

            
            if (cur_shift == 0):
                result = splited
            else:   
                result = np.concatenate((result, splited))
            #pectrogramm = np.array(spectrogramm[cur_shift:])
        
        #print(np.array(result).shape)
        #print(self.getClassifiedFragment([["1", "51515"]], "225"))
        #print(self.getClassifiedFragment(result, audio_class))
        return result 

    # def pipelieFile(self):
    #     """DEPRECATED"""
    #     #audioClass = getAudioClass(file_path)
    #     self.__getNumpyedAudio()
    #     #print(self.__PCM)
    #     self.__getPCMFragments()

    #    # random_id = str(random.random())
    #     counter = 0
    #     start = time.perf_counter()

    #     #self.__saveMelSpectrogram(S, name)
    #     for fragment in self.__fragments:        

    #         S = self.__getMelSpectrogram(fragment)

    #     ## calculate index in fragments consistently with order of default PCM, i.e. frame by frame
    #     ## it needs because fragments doesnt have sorted in order of directrly location
    #         index = (counter % self.__fragments_number) * self.__layering + counter / self.__fragments_number
    #         name = self.__out_spectr_dir + "\\" + self.__out_spectr_file + str(index) + ".png"
    #         #print(name)
    #         self.__saveMelSpectrogram(S, name)

    #         counter = counter + 1

    #     #print(time.perf_counter() - start)

    def getAudoName(self):
        return os.path.basename(self.__path)

    def saveToDir(self, audio_class):
        """Saves current audio file as spectrograms with specified parameters 
        to the appropriate folder

        Parameters
        ----------
        audio_class : str
            name of class (folder to save to)

        Returns
        -------
        None
        """
        st = time.perf_counter()
        specs = self.getSpectrogramsArray()
        
        #print("Fragment number: ", len(specs))
        #print(len(specs))
        uniqueId = str(random.randint(0, sys.maxsize))
        for i in range(len(specs)):
            path = ""
            if audio_class is None:
                path = self.__out_spectr_dir + os.sep + uniqueId + "_" + str(i)
            else:
                path = self.__out_spectr_dir + os.sep + audio_class + os.sep + uniqueId + "_" + str(i)
            self.__saveMelSpectrogram(specs[i], path)
        
        print("#\nConverting ", self.getAudoName(), "\nSave time: ", time.perf_counter() - st, flush=True)
        
# f = File(r"J:\Jupyter\Jupyter\data\SOD\sample1.mp3", 
#          r"J:\Jupyter\Jupyter\specs", "full", fragment_size_ms=30000, n_fft=2048, layering=2)
# f.saveToDir("SOD")
