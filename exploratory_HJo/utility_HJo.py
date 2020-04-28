# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 02:49:10 2020

@author: HyeongChan Jo
"""
import numpy as np

def dataPrep_LSTM_fromDF(dataList, targetName, input_size, output_size, step_size, testingSet_size):
    ## inputs
    ##  - dataList: list of dataframes for training/testing the model
    ##  - targetName: name of the column to be used for prediction
    ##  - input_size: size of the input data
    ##  - output_size: size of the output data (i.e. horizon)
    ##  - step_size: step size between each of training dataset
    ##  - testingSet_size: size of the validation set, in terms of the number of outputs
    
    # initialization
    trainingData = []
    trainingAns = []
    testingData = []
    testingAns = []
    
    for df in dataList:
        for i in range(input_size, len(df)-testingSet_size-output_size, step_size):
            trainingData.append(np.array(df[i-input_size : i, :]))
            trainingAns.append(np.array(df[i : i+output_size, targetName]))
        
        for i in range(len(df)-testingSet_size, len(df)-output_size, step_size):
            testingData.append(np.array(df[i-input_size : i, :]))
            testingAns.append(np.array(df[i : i+output_size, targetName]))
            
    return np.array(trainingData), np.array(trainingAns), np.array(testingData), np.array(testingAns)


def dataPrep_LSTM_fromNP(dataList, targetName, input_size, output_size, step_size, testingSet_size, columnName):
    ## inputs
    ##  - dataList: list of dataframes for training/testing the model
    ##  - targetName: name of the column to be used for prediction
    ##  - input_size: size of the input data
    ##  - output_size: size of the output data (i.e. horizon)
    ##  - step_size: step size between each of training dataset
    ##  - testingSet_size: size of the validation set, in terms of the number of outputs
    ##  - columnName: name of each column
    
    # initialization
    trainingData = []
    trainingAns = []
    testingData = []
    testingAns = []
    
    # find the target column
    targetIdx = columnName.index(targetName)
    
    for df in dataList:
        for i in range(input_size, len(df)-testingSet_size-output_size, step_size):
            trainingData.append( df[i-input_size : i, :] )
            trainingAns.append( df[i : i+output_size, targetIdx] )
        
        for i in range(len(df)-testingSet_size, len(df)-output_size, step_size):
            testingData.append( df[i-input_size : i, :] )
            testingAns.append( df[i : i+output_size, targetIdx] )
            
    return np.array(trainingData), np.array(trainingAns), np.array(testingData), np.array(testingAns)