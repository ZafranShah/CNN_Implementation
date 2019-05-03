# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 15:27:41 2019

@author: shah
"""
import os
import keras
from JsonParameters import ConfigurationParameters
from DataPreprocessing import DataPreparation
import logging, argparse
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D



################# Logging ##########################################

logger=logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter=logging.Formatter('%(asctime)s | %(levelname)s -> %(message)s')


##########################Declaration & Training of CNN model ###################################

def CreateModel(Img_size, Image_chanel):
    model = Sequential()
    model.add(Convolution2D(32,(3,3), padding='same', input_shape=(Img_size,Img_size,Image_chanel)))
    model.add(Activation('relu'))
    model.add(Convolution2D(32,(3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(totalClasses))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer ='adam', metrics =['accuracy'])
    model.fit(train_x[:100], train_y[:100], batch_size=param.batchSize, epochs=param.epochs, verbose=1, validation_data =(valid_x[:10], valid_y[:10]))
    return model


########################### Saving Trained Model & Weights to Disk ###############
def SavingModel(save_path,model_name):
    save_Path = os.path.join(os.getcwd(), save_path)
    model_Name = model_name
    if not os.path.isdir(save_Path):
        os.makedirs(save_Path)
        model_path = os.path.join(save_Path, model_Name)
        model1.save(model_path)
        logger.info('Saved trained model at %s :',model_path)
    else:
        model_path = os.path.join(save_Path, model_Name)
        model1.save(model_path)
        logger.info('Saved trained model at %s :',model_path)




def Main():
    parser= argparse.ArgumentParser()

    parser.add_argument('PathofJsonFile', help= 'Enter the of the configuration file')   
    args = parser.parse_args()
    return args

###############Parsing Command Line Arguement############
    
arguements=Main()
param= ConfigurationParameters(arguements.PathofJsonFile)


param.AssignParameters()
#######################Logging########################
file_handler=logging.FileHandler(param.logFile)
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.DEBUG)
logger.addHandler(file_handler) 

datapreprocessing=DataPreparation(param.img_Dir,param.img_Size, param.augmentation, param.imgRotation, param.augImgNumber,param.pathtoAugImg, param.pathtoSaveAugImg, param.batchSize,param.pathtoTrainingData, param.augImgFlip)
       
        
        
#### Loading Data and preprocessing of Training , Validation and Test Data #################

train_x, train_y = datapreprocessing.LoadPickledData(param.pathtoTrainingData)
valid_x, valid_y = datapreprocessing.LoadPickledData(param.pathtoTestData)

logger.info('Training and test data is loaded')

################ Training Model##############################
totalClasses= datapreprocessing.TotalClasses(train_y)
train_y=keras.utils.to_categorical(train_y, num_classes=totalClasses)
valid_y=keras.utils.to_categorical(valid_y, num_classes=totalClasses)

logger.info('Total number of classes in the data are %d:',totalClasses)
        
model1=CreateModel(param.img_Size, param.imgChannel)

######################Saving Weights #########################
 

SavingModel(param.pathtoSaveTrainModel, param.trainedModelName)
#########################Main Entry Point##################

if __name__ == "__main__":
   
  Main()

