# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 16:34:53 2019

@author: shah
"""
import logging,json



class ConfigurationParameters:
    logFile='\\log.txt'
    img_Dir=''
    img_Size=32
    augmentation= False
    imgRotation=0
    augImgNumber=0
    pathtoAugImg=''
    pathtoSaveAugImg=''
    dataGenBatchSize=0
    augImgFlip = False
    picklingData= False
    pathtoPickledTrainingData=''
    pathtoPickledTestData=''
    pathtoTrainingData=''
    imgChannel=1
    epochs=10
    batchSize=10
    pathtoSaveTrainModel=''
    trainedModelName= ''
    pathtoTestData=''
    pathtoConfusionMatrix=''
    def __init__(self,PathofJsonFile ):
        
        self.path = PathofJsonFile
        
    def LoadParameters(self):
        if self.path.strip():
            with open(self.path) as paramfile:
                data=json.load(paramfile)
                return data
        else:
            logger.error('please load the Json File that contains the configuration parameters')
    
    def AssignParameters(this):
        JsonParameters=this.LoadParameters()
        if len(JsonParameters) is not 0:
            logger.info('parsing paramter from config file')
            this.logFile=JsonParameters['Data_Preprocessing']['PathToLogFile']
            this.img_Dir=JsonParameters['Data_Preprocessing']['ImageDirectory']
            this.img_Size=JsonParameters['Data_Preprocessing']['SizeofImage']
            this.augmentation=JsonParameters['Data_Preprocessing']['Image_Augmentation']['Augmentation']
            this.imgRotation=JsonParameters['Data_Preprocessing']['Image_Augmentation']['Image_Rotation']
            this.augImgNumber=JsonParameters['Data_Preprocessing']['Image_Augmentation']['NumberofImages']
            this.pathtoAugImg=JsonParameters['Data_Preprocessing']['Image_Augmentation']['PathofImageForAugmentation']
            this.pathtoSaveAugImg=JsonParameters['Data_Preprocessing']['Image_Augmentation']['PathforAugImages']
            this.dataGenBatchSize=JsonParameters['Data_Preprocessing']['Image_Augmentation']['DataGene_BatchSize']
            this.augImgFlip=JsonParameters['Data_Preprocessing']['Image_Augmentation']['Image_Flip']
            this.picklingData=JsonParameters['Data_Preprocessing']['PicklingData']
            this.pathtopickledTrainingData=JsonParameters['Data_Preprocessing']['SavingPickledTrainData']
            this.pathtoPickledTestData=JsonParameters['Data_Preprocessing']['SavingPickledTestData']
            this.pathtoTrainingData=JsonParameters['CNN_Train_Model']['PathtoTrainingData']
            this.imgChannel=JsonParameters['CNN_Train_Model']['ImageChannel']
            this.epochs=JsonParameters['CNN_Train_Model']['Epochs']
            this.batchSize=JsonParameters['CNN_Train_Model']['Batchsize']
            this.pathtoSaveTrainModel=JsonParameters['CNN_Train_Model']['PathToSaveModels']
            this.trainedModelName=JsonParameters['CNN_Train_Model']['TrainedCNNModel']
            this.pathtoTestData=JsonParameters['CNN_Test_Model']['PathofTestData']
            this.pathtoConfusionMatrix=JsonParameters['CNN_Test_Model']['PathtoSaveConfidenceMatrix']
        else:
            logger.error('The configuration file doesnot contain any parameter or incorrect path')
            return 0



logger=logging.getLogger(__name__)
logger.info('Parameters are successfully loaded from the Json file')
