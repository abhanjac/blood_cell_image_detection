# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 08:52:32 2019

@author: abhanjac
"""

import tensorflow as tf

from utils_2 import *
from tiny_yolo_classifier_2 import *

#===============================================================================

if __name__ == '__main__':
    
    trainDir = 'train'
    validDir = 'valid'
    testDir = 'test'
    trialDir = 'trial'
    
    #tyClassifier = tinyYolo1()
    #tyClassifier.train( trainDir=trainDir, validDir=validDir )
##    tyClassifier.train( trainDir=trialDir, validDir=trialDir )


    tyDetector = tinyYolo2()
    tyDetector.train( trainDir=trainDir, validDir=validDir )
##    tyDetector.train( trainDir=trialDir, validDir=trialDir )
    