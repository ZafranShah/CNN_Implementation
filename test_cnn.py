# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 17:23:14 2019

@author: shah
"""
import keras
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import scikitplot as skplt
from DataPreprocessing import DataPreparation
from JsonParameters import ConfigurationParameters


###############Loading Trained Model and Applying Test Data###############################

def LoadTrainedModel(pathofTrainModel):
    model = load_model(pathofTrainModel)
    pred_y=model.predict(test_x)
    scores=model.evaluate(test_x, test_y, verbose=1)
    return pred_y, scores

##############Ploting and Saving Confusion Matrix #######################################
def PlotandSaveConfidenceMatrix(filetype,testy, predy):
    predy= np.argmax(predy, axis=1)
    testy=np.argmax(testy, axis=1)
    if len(testy) ==len(predy):
        skplt.metrics.plot_confusion_matrix(testy, predy, normalize=False, figsize=(15,15 ))
        return plt.savefig(str(filetype))


##############parsing parameters from configuration file ###########################
param= ConfigurationParameters("../../configuration.json")
param.AssignParameters()
datapreprocessing=DataPreparation(param.img_Dir,param.img_Size, param.augmentation, param.imgRotation, param.augImgNumber,param.pathtoAugImg, param.pathtoSaveAugImg, param.batchSize,param.pathtoTestData, param.augImgFlip)

##############Loading Test Data ######################################
test_x, test_y = datapreprocessing.LoadPickledData(param.pathtoTestData)

totalClasses= datapreprocessing.TotalClasses(test_y)
print ('total classes are:', totalClasses)
test_y=keras.utils.to_categorical(test_y, num_classes=totalClasses)

##################Evaluating Trained Model on Test Data######################      
prediction, Scores = LoadTrainedModel(param.trainedModelName)
print('Test loss:', Scores[0])
print('Test accuracy:', Scores[1])    

###############Saving Confusion Matrix ########################################
    
PlotandSaveConfidenceMatrix(param.pathtoConfusionMatrix,test_y, prediction)