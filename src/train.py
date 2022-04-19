# Copyright (c) 2021, Pavel Alexeev, pavlik3312@gmail.com
# All rights reserved.
#
# This source code is licensed under the CC BY-NC-SA 4.0 license found in the
# LICENSE file in the root directory of this source tree.



#import .utility as utility
from sklearn.metrics import precision_score
from keras import callbacks
import model as models
import preprocessor as process_files

import random

import tensorflow as tf
import pathlib

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam 

# TODO: make a classification report

from keras.models import Sequential, load_model

######################### utils ###########################
def checkPathAndCreate(path):
    if (not os.path.exists(path)):
        os.makedirs(path)

def getImageShape(path):
    return tf.image.decode_png(tf.io.read_file(path)).shape


def getPrecision(confusion_matrix):        
    size = confusion_matrix.shape[0]

    # some classes in confusion matrix could be not presented
    # it wont count as existed class
    real_size = 0
    precision_sum = 0
    for i in range(0,size):
        true_res = confusion_matrix[i][i]
        all_res = sum(confusion_matrix[i])
        if (all_res > 0):
            precision_sum += true_res/all_res
            real_size += 1
    return precision_sum / real_size
        
def getRecall(confusion_matrix): 
    confusion_matrix = np.transpose(confusion_matrix)

    size = confusion_matrix.shape[0]

    return getPrecision(confusion_matrix)

def getF1(precision, recall, beta=2):
    b_squared = beta*beta
    return (b_squared + 1) * (precision * recall) / (b_squared * precision + recall)

def getAccuracy(confusion_matrix):
    size = confusion_matrix.shape[0]

    sum = tf.reduce_sum(confusion_matrix).numpy()
    true_results = 0
    for i in range(0, size):
        true_results += confusion_matrix[i][i]
    
    return true_results / sum

def getMetrics(confusion_matrix):
    precision = getPrecision(confusion_matrix)
    recall = getRecall(confusion_matrix)
    f1 = getF1(precision, recall)
    accuracy = getAccuracy(confusion_matrix)
    return precision, recall, f1, accuracy
###########################################################

    
## processing spectrogram shape
#INPUT_SHAPE_IMG = False

class Train:
    """
    Train 
    """
    ## class parameters are only work constants 
    ## whitch depends on current dataset
    ## another patameters pass throught START() funct

    RANDOM_SEED = 777

    classesNumber = 0
    datasetSize = 0
    imageShape = False

    # "add paths manager"

    def __init__(self, seed=777):
        self.RANDOM_SEED = seed

    def getDataset(self, spectrogram_out_dir, batch_size):
        """Form dataset from previous transfered classified spectrograms.

        Parameters
        ----------
        spectrogram_out_dir : str
            path to directory with classified spectrograms

        batch_size : str
            size for batch normalization

        Returns
        -------
        train_dataset, val_dataset, test_dataset : tf.data.Dataset
            shuffled datasets

        label_names : dict
            numbered labels of classes

        INPUT_SHAPE_IMG
            spectrogram image shape

        """
        
        ## this directory must be empty
        #none_class = spectrogram_out_dir + os.sep + "_NONE_"
        #checkPathAndCreate(none_class)

        ## open and prepare spectrogram dataset
        data_root = pathlib.Path(spectrogram_out_dir)

        all_image_paths = list(data_root.glob('*/*'))
        all_image_paths = [str(path) for path in all_image_paths]
        all_image_paths = all_image_paths
        ## shuffling all images with seed
        random.Random(self.RANDOM_SEED).shuffle(all_image_paths)

        DATASET_SIZE = len(all_image_paths)

        ## get all classes
        label_names = sorted(item.name for item in data_root.glob('*/') 
                             if item.is_dir())
        print(label_names, flush=True)


        ## setup number of classes
        self.classesNumber = len(label_names)

        ## indexing classes
        label_to_index = dict((name, index) for index, name in enumerate(label_names))
        print(label_to_index, flush=True)

        all_image_labels = [label_to_index[pathlib.Path(path).parent.name]
                            for path in all_image_paths]    

        ################################# to utils #####################################
        def genResult(pos, len):
            res = [0. for i in range(len)]
            res[pos] = 1.
            return res
        ################################################################################

        lableNum = len(label_names)

        all_image_labels = [genResult(i,lableNum) for i in all_image_labels]
        #print(all_image_labels[1]) 

        ## set input shape 
        INPUT_SHAPE_IMG = getImageShape(all_image_paths[0])
        PIXEL_SHAPE = (INPUT_SHAPE_IMG[0], INPUT_SHAPE_IMG[1])
        #print(PIXEL_SHAPE)
        
        ## dataset prepared
        ## preprocess functions
        def preprocess_image(image):
            image = tf.image.decode_png(image, channels=3)
            image = tf.image.resize(image, PIXEL_SHAPE)
            #print("Image Shape:", image.shape)
            # normalize to [0,1] range
            image = tf.cast(image, tf.float32) / 255.0  
            return image

        def load_and_preprocess_image(path):
            #print(path)
            image = tf.io.read_file(path)
            #print(image.shape)
            return preprocess_image(image)

        load_and_preprocess_image(all_image_paths[1])

        ## create Dataset
        print("All images: ", len(all_image_paths), flush=True)
        ds = tf.data.Dataset.from_tensor_slices((all_image_paths, 
                                                 all_image_labels))

        def load_and_preprocess_from_path_label(path, label):
            return load_and_preprocess_image(path), label

        ### WARNING: MAP DOESNT PASS WITHOUT CONSTANT SHAPE OF IMAGE IN PROCESS 
        ### IMAGE FUNCTION
        image_label_ds = ds.map(load_and_preprocess_from_path_label)
        print(image_label_ds, flush=True)

        
        # shape of processed image
        INPUT_SHAPE_IMG = image_label_ds.element_spec[0].shape

        image_label_ds = image_label_ds.batch(batch_size)
        #image_label_ds = image_label_ds.prefetch(1)

        
        train_size = int(0.70 * DATASET_SIZE / np.float(batch_size))
        val_size   = int(0.15 * DATASET_SIZE / np.float(batch_size))
        test_size  = int(0.15 * DATASET_SIZE / np.float(batch_size))
        print("DATA_SIZE: ", DATASET_SIZE, flush=True)
        print("BATCH_SIZE: ", np.float(batch_size), flush=True)
        #print(0.70 * DATASET_SIZE / np.float(batch_size))
        print("train_size: ", train_size, flush=True)
        print("val_size: ", val_size, flush=True)
        print("test_size: ",test_size, flush=True)

        train_dataset = image_label_ds.take(train_size)
        test_dataset = image_label_ds.skip(train_size)
        val_dataset = test_dataset.take(val_size)
        test_dataset  = test_dataset.skip(val_size)
        # print("train_dataset size: ", tf.size(train_dataset))
        # print("test_dataset size: ",  tf.size(test_dataset))
        # print("val_dataset size: ",   tf.size(val_dataset))
        return train_dataset, val_dataset, test_dataset, \
               label_names, INPUT_SHAPE_IMG

    def prepareModel(self, image_shape, learning_rate):
        """Compile model for current image shape

        Parameters
        ----------
        image_shape : np.ndarray.shape
            `shape` of image with color channels

        learning_rate : float

        Returns
        -------
        model : keras.Model
            compiled neural network
        """

        print('IMAGE SHAPE = ', image_shape, flush=True)
        print('CLASSES NUMBER = ', self.classesNumber, flush=True)
        model = models.CRNN2D(image_shape, nb_classes = self.classesNumber)
        model.compile(loss = 'categorical_crossentropy',
                      optimizer = Adam(lr=learning_rate),
                      metrics = ['accuracy'])
        model.summary()
        return model

    def setCallbacks(self, checkpoints_path, is_early_stop, patience=10):
        """Create and choose callbacks for model. It could be `checkpoint only` 
        or `chechpoint and earlystop` to save time for training

        Parameters
        ----------
        checkpoints_path : str
            path to saving directory
        
        is_early_stop : bool
            flag to control stop before all epoches

        Returns
        -------
        callbacks : list of keras.callbacks
        """
        checkpointer = ModelCheckpoint(filepath=checkpoints_path,
                                       verbose=1,
                                       save_best_only=True)
        earlystopper = EarlyStopping(monitor='loss', min_delta=1,
                                     patience=patience, verbose=0, mode='auto')
        callbacks = [checkpointer]
        if (is_early_stop):
            callbacks.append(earlystopper)
        return callbacks

    def trainModel(self, model, train, validation, epochs, batch_size, 
                   callbacks, save_weights_folder, save_name):
        """Training model by `train` and `validation` dataset, given number of 
        `epoches`, `batch size`, `callbacks`. After training model saves 

        Parameters
        ----------
        train, validation : tf.data.Dataset
            batched datasets

        epochs : int 
            number of train epoches

        batch_size : int
            size for batch normalization 

        list of keras.callbacks

        save_weights_folder : str
            folder where the file is saved

        save_name : str
            name to save the model as file

        Returns 
        -------
        history : History 
            metrics of training process and it results
        """
        
        history = model.fit(train, 
                            validation_data = validation, 
                            epochs = epochs, 
                            batch_size = batch_size,                      
                            use_multiprocessing = True, 
                            callbacks = callbacks)
        ## path manager ##
        weights = os.path.join(save_weights_folder, save_name)
        checkPathAndCreate(weights)        
        ##################
        model.save(weights + os.sep + save_name + ".h5")
        # load_model(r"J:\Jupyter\Jupyter\weights")
        return history

    def testModel(self, model, test, batch_size):
        """
        Parameters
        ----------
        model : keras.Model
            already trained model

        test : tf.data.Dataset
            test dataset

        batch_size : int
            size for batch normalization

        Returns 
        -------
        result : list of list of float
            distribution of predicted probabilities
        """
        result = model.predict(test, batch_size = batch_size)
        return result
        

    def processResult(self, test_dataset, result):
        prediction = np.argmax(result, axis=1)
        #print(pred)
        y_true = test_dataset.map(lambda img, label: 
                                  tf.math.argmax(label, axis=1))
        y_true = y_true.flat_map(lambda x: 
                                 tf.data.Dataset.from_tensor_slices(x))
        #print(y_true)
        y_true = np.fromiter(y_true.as_numpy_iterator(), dtype = int)
        # print(type(y_true))
        # for elem in y_true.as_numpy_iterator():
        #     print(elem)

        #print(y_true)
        return prediction, y_true





    def saveMetrics(self, save_metrics_folder, save_name, predictions,
                    true_results, label_names, history = 0): 
        save_metrics_folder += os.sep + save_name
        checkPathAndCreate(save_metrics_folder)

        confusion_mtx = tf.math.confusion_matrix(labels = true_results, 
                                                  predictions = predictions)


        # precision = getPrecision(confusion_mtx)
        # recall = getRecall(confusion_mtx)
        # f1 = getF1(precision, recall)
        # test_acc = sum(predictions == true_results) / len(true_results)

        precision, recall, f1, accuracy = getMetrics(confusion_mtx)

        print(f'Test set precision: {precision:.4%}', flush=True)
        print(f'Test set recall: {recall:.4%}')
        print(f'Test set f1: {f1:.4%}')
        print(f'Test set accuracy: {accuracy:.4%}')
        
        # metrix file output
        f = open(save_metrics_folder+os.sep+save_name+"metric_results.txt", 'w')
        f.write(f'Test set precision: {precision:.4%}\n')
        f.write(f'Test set recall   : {recall:.4%}\n')
        f.write(f'Test set f1       : {f1:.4%}\n')
        f.write(f'Test set accuracy : {accuracy:.4%}\n')
        f.close()


        plt.figure(figsize=(10, 8))
        sns.heatmap(confusion_mtx, xticklabels=label_names, 
                    yticklabels=label_names, annot=True, fmt='g')
        plt.xlabel('Prediction')
        plt.ylabel('Label')
        #plt.show()


        plt.title("Heatmap ("+str(len(true_results))+
                  " elements)\n Accuracy "+str(accuracy)+"%")
        plt.savefig(save_metrics_folder+os.sep+
                    save_name + "_confusion_mtrx" + '.png') 
        plt.close()

        # summarize history for accuracy 
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model accuracy (total '+
                  str(history.history['accuracy'][-1]) + ')')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(save_metrics_folder+os.sep+
                    save_name+"_model_accuracy"+'.png') 
        plt.close()

        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(save_metrics_folder+os.sep+
                    save_name + "_model_loss" + '.png') 
        plt.close()

    def startTrain(self, 
              spectrogram_out_dir : str,
              save_metrics_folder : str,
              save_weights_folder : str,    
              
              save_name : str,

              batch_size : np.float,
              epochs : int,
              early_stop : bool,
              patience : int,
              learning_rate : np.float):
        """ 
        
        """        
        print("Creatin datasets")
        train, validation, test, \
        self.labelNames, self.imageShape = self.getDataset(spectrogram_out_dir,
                                                           batch_size)
        # print("Loading model")
        # model = load_model(save_weights_folder+os.sep+save_name+os.sep+
        #             save_name+".h5")
        # model.summary()
        model = self.prepareModel(self.imageShape, learning_rate)
        callbacks = self.setCallbacks(save_weights_folder, early_stop, patience)        

        history = self.trainModel(model, train, validation, epochs, batch_size, 
                                  callbacks, save_weights_folder, save_name)   
        test_result = self.testModel(model, test, batch_size)
        predictions, true_results = self.processResult(test, test_result)
        
        self.saveMetrics(save_metrics_folder, save_name, predictions, 
                         true_results, self.labelNames, history)

        print("Testing")
        test = train.concatenate(validation.concatenate(test))
        print("Test started")
        test_result = self.testModel(model, test, batch_size)
        print("Printing to results")
        predictions, true_results = self.processResult(test, test_result)
        self.saveMetrics(save_metrics_folder, save_name+"alldata", predictions, 
                          true_results, self.labelNames)

        return self.labelNames
        