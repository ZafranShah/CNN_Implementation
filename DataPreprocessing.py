# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 12:41:39 2019

@author: shah
"""

import os 
import cv2
import random
import numpy as np 
import logging
import pickle, gzip
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

Dataset=[]
classes=[]

class DataPreparation():
    
    def __init__(self, Data_Path, Img_Size, Image_Augmentation,  Img_Rotation, Augmented_Img, Image_path, Directory, Batch_Size, Pickledatapath ,Image_flip=False):
        
        self.dataPath = Data_Path
        self.imgSize= Img_Size
        self.imgRotation= Img_Rotation
        self.numberofAugmentedImg= Augmented_Img
        self.imgForAugmentation=Image_path
        self.dir= Directory
        self.batchSize=Batch_Size
        self.imgFlipHorizontal= Image_flip
        self.imageAugmentation = Image_Augmentation
        self.filename=Pickledatapath
    
    def DataAugmentation(self):
        if self.imageAugmentation is True:
            logger.info('Data Augmentation method is called from the preprocessing module')
            datagenerator = ImageDataGenerator(rotation_range=self.imgRotation, width_shift_range=0.2, 
                                               height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=self.imgFlipHorizontal, fill_mode='nearest')
            logger.info('Image loaded from', self.imgForAugmentation)
            logger.info('Image flipped from', self.imgFlipHorizontal)
            img = load_img(self.imgForAugmentation)
            x=img_to_array(img)
            x=x.reshape((1,) + x.shape)
            i = 0
            for batch in datagenerator.flow(x, batch_size=self.batchSize, save_to_dir =self.dir):
                i += 1
                if i >self.numberofAugmentedImg:
                    break
        else:
            logger.error('please check Image_Augmentation parameter in the configuration file or the path of the augmented image')
    def DataPreprocessing(self):
        logger.info('Data preprocessing method is called')
        if self.dataPath.strip():
            for labels in os.listdir(self.dataPath):
                classes.append(labels)
            for clas in classes:
                path = os.path.join(self.dataPath, clas)
                class_num=classes.index(clas)
                for image in os.listdir(path):
                    try:
                        img_array= cv2.imread(os.path.join(path, image))
                        new_array = cv2.resize(img_array,(self.imgSize,self.imgSize))
                        Dataset.append([new_array,class_num])                    
                    except Exception as e:
                        logger.error(e)
                        pass
        else:
            logger.error('please provide the path of the directory which contain images')
            return 0
        logger.info('Data Extracted from the Directories')
        images=[]
        labels=[]
        random.shuffle(Dataset)
        if len(Dataset) is not 0:
            for feature, label in Dataset:
                images.append(feature)
                labels.append(label)
        else:
            logger.error('No data found please recheck the directory')
        logger.info('Separating data in different patches')
#    xtrain, xtest, ytrain, ytest = train_test_split(images, labels, test_size=0.20, random_state=42)
        xtrain, xtest, ytrain, ytest = train_test_split(images,labels, test_size=0.20,random_state=0, stratify=labels)
        logger.info('Data Normalization and Scaling')
        xtrain = np.array(xtrain, dtype=np.float32)
        xtest = np.array(xtest, dtype=np.float32)
        xtrain /= 255
        xtest /= 255 
        print (len(xtrain))
        return  xtrain, xtest, ytrain, ytest, classes
    def TotalClasses(self, arg1):
        totalLabels=[]
        if len(arg1) is not 0:
            for x in arg1:
                if x not in totalLabels:
                    totalLabels.append(x)
        return len(totalLabels)
    
    def PicklingData(self):
        Dataset.clear()
        classes.clear()
        classno.clear()
        logger.info('Pickling the data into training and test file')
        train_x, test_x, train_y, test_y, originalclass=self.DataPreprocessing()
        self.training_data= train_x, train_y
        self.test_data= test_x, test_y
        if len(self.training_data) is not 0:
            logger.info('Saving the Preprocessed training data in the directory')
            Trainingfilename = 'trainingdata1.pkl.gz'
            with gzip.open(Trainingfilename, 'wb') as outfile:
                pickle.dump(self.training_data,outfile)
                outfile.close()
        if len(self.test_data) is not 0:
            logger.info('Saving the Preprocessed test data in the directory')
            testfilename = 'testdata1.pkl.gz'
            with gzip.open(testfilename, 'wb') as outfile1:
                pickle.dump(self.test_data,outfile1)
                outfile1.close()
        else:
            logger.error('No training or test data found.... Nothing to Pickle')
            
            
    def LoadPickledData(self, arg1):
        logger.info('Loading data from the Directory in the pickled format')
        with gzip.open(arg1,'rb') as infile:
            data = pickle.load(infile)
            features, labels = data
            infile.close()
            return features, labels 

        
        






logger=logging.getLogger(__name__)



