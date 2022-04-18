# music-artist-classifier
Project is a full music artistist classifier, using **Tensorflow** and Keras. Algorithm can process almost all music formats and can fully work with configuration file from music files to needed results. 

Algorithm classifies audio files by artist, which tracks have already been known. To classify files probable known artist are needed to be in training dataset.

## Requirenments
- [ffmpeg](https://ffmpeg.org/ "Official site")
- python 3.8.x or more (campatible with Tensorflow 2.8 or more)
	- tensorflow 2.8
	- numpy
	- etc. (see requirenments.txt)
- [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) (for using Tensorflow)
- free space to generate temp files
## Start

To start project you need to copy ```src``` folder, set up config file (manually or with *Configuratoin Editor*) and start **main_config.py** in console:
```
C:\\...\src>main_config.py -i config_path\config.ini -m all|prepare|train|work
```
You have to check especialy folder srtucture (see *\example* folder) and config file. 

## Directory system
### Dataset
Dataset must contain different folders , representing classes (author, artists) of files in them. Only first depth directories have to get strict structure, further directories could have any files and subdirectories. Files to be out according to file format that can be configured.
### Work 
Work folder represents files to classify. Files must be directly be in work folder, without any subfolders.
Missing folders are going to be created. 

Project source files and dataset directory system should be in different dirrectories to avoid confusoins with files. Algorithm generates a lot of new image files that need in processing (it can be removed mannualy after algorithm stops).


