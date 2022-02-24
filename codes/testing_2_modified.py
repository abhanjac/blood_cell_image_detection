# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 14:28:52 2019

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

##-------------------------------------------------------------------------------
#    
#    tyClassifier.saveLayer = True   # Activate the flag to save the layer outputs.
#    inferDir = trialDir
#    key = '`'
#    listOfImg = os.listdir( os.path.join( inferDir, 'images' ) )
#    listOfImg.sort()
#    nImgs = len( listOfImg )
#
###-------------------------------------------------------------------------------
##    
##    # Result on the misclassified images.
##    with open( 'misclassified_example_list.json', 'r' ) as infoFile:
##        inferDir = testDir
##        infoDict = json.load( infoFile )
##        listOfImg = list( infoDict )
##        
##-------------------------------------------------------------------------------
#
#    for idx, i in enumerate( listOfImg ):
#        labelDictList, multiHotLabel = getImgLabel( inferDir, i )
#        multiHotLabel = multiHotLabel.tolist()
#        
##        # Skip images if needed.
##        if idx < 23:   continue
#        
#        # Prediction from network.
#        img = cv2.imread( os.path.join( inferDir, 'images', i ) )
#        img1 = copy.deepcopy( img )
#        imgBatch = np.array( [ img ] )
#        inferLayerOut, inferPredLogits, inferPredProb, inferPredLabel, _, _ = \
#                                            tyClassifier.batchInference( imgBatch )
#        inferPredLabel = inferPredLabel.tolist()[0]
#
#        isPredictionCorrect = True if multiHotLabel == inferPredLabel else False
#
##-------------------------------------------------------------------------------
#
#        # Printing the TRUE and PREDICTED labels.
#        print( '\n{}/{}'.format( idx+1, nImgs ) )
#        print( 'True multi-hot label: {}'.format( multiHotLabel ) )
#        print( 'Predicted multi-hot label: {}'.format( inferPredLabel ) )
#        print( '{} prediction'.format( isPredictionCorrect ) )
#        
#        # Displaying ground truth class names.
#        k = 0
#        for jdx, j in enumerate( multiHotLabel ):
#            if j == 1:
#                k += 1
#                className = classIdxToName[ jdx ]
#                print( 'True Label: {}'.format( className ) )
#                cv2.putText( img, className, (10,int(20*(k+1))), \
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2, cv2.LINE_AA )
#        
#        # Displaying predicted class names.
#        k = 0
#        for jdx, j in enumerate( inferPredLabel ):
#            # If the prediction label element matches the corresponding true 
#            # label element, then print the predicted label name in green, else
#            # print in red.
#            color = (0,255,0) if j == multiHotLabel[ jdx ] else (0,0,255)
#            
#            if j == 1:
#                k += 1
#                predName = classIdxToName[ jdx ]
#                print( 'Predicted Label: {}'.format( predName ) )
#                cv2.putText( img, predName, (10,120+int(20*(k+1))), \
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA )
#
##-------------------------------------------------------------------------------
#
#        # Display the last gap layer if the layers are saved.
#        if tyClassifier.saveLayer:
##            print( list( inferLayerOut ) )
#            layer = inferLayerOut['conv19']
#            b, h, w, nChan = layer.shape
#            
#            # Stacking the channels of this layer together for displaying.
#            for c in range( nChan ):
#                channel = layer[0,:,:,c]
#                resizedChan = cv2.resize( channel, (inImgW, inImgW), \
#                                             interpolation=cv2.INTER_LINEAR )
#                minVal, maxVal = np.amin( resizedChan ), np.amax( resizedChan )
#
#                # Normalizing the output and scaling to 0 to 255 range.
#                normalizedChan = ( resizedChan - minVal ) / ( maxVal - minVal \
#                                                              + 0.000001 ) * 255
#                normalizedChan = np.asarray( normalizedChan, dtype=np.uint8 )
#                
#                # Stacking the normalized channels.
#                if c == 0:
#                    layerImg1 = normalizedChan
#                elif c > 0 and c < int( np.ceil( nClasses / 2 ) ):
#                    layerImg1 = np.hstack( ( layerImg1, normalizedChan ) )
#                
#                if c == int( np.ceil( nClasses / 2 ) ):
#                    layerImg2 = normalizedChan
#                elif c > int( np.ceil( nClasses / 2 ) ) and c < nClasses:
#                    layerImg2 = np.hstack( ( layerImg2, normalizedChan ) )
#                    
##-------------------------------------------------------------------------------
#                    
#        if layerImg1.shape != layerImg2.shape:  # May happen if nClasses is odd.
#            blankImg = np.zeros_like( normalizedChan )
#            layerImg2 = np.hstack( ( layerImg2, blankImg ) )
#            
#        layerImg = np.vstack( ( layerImg1, layerImg2 ) )                    
##        cv2.imshow( 'Gap layer', layerImg )                        
#                                        
##-------------------------------------------------------------------------------
#                
#        cv2.imshow( 'Image', img )
#        print( i )
#        key = cv2.waitKey(1)
#        if key & 0xFF == 27:    break    # break with esc key.
#    
#    cv2.destroyAllWindows()
#    
    
    
    
##===============================================================================
#
    tyClassifier = tinyYolo2()
    tyClassifier.test( testDir=testDir )
    tyClassifier.calcRBCperformance( testDir )
    tyClassifier.test( testDir=validDir )
    tyClassifier.calcRBCperformance( validDir )
#    tyClassifier.test( testDir=trainDir )
#    tyClassifier.calcRBCperformance( trainDir )
#
##-------------------------------------------------------------------------------

    #tyClassifier.saveLayer = True   # Activate the flag to save the layer outputs.
    #key = '`'
    
##-------------------------------------------------------------------------------

    #inferDir = testDir
##    inferDir = 'presentation_images'
    #listOfImg = os.listdir( os.path.join( inferDir, 'images' ) )
    #listOfImg.sort()
    #np.random.shuffle( listOfImg )    # Shuffling the list randomly.
    #nImgs = len( listOfImg )
    
    ## We also need to calculate the precision and recall between the infected 
    ## and normal rbc cells.
    #rbcTP, rbcFP, rbcTN, rbcFN = 0.0, 0.0, 0.0, 0.0
    
    #for idx, i in enumerate( listOfImg ):
        #startTime = time.time()
        
        #labelDictList, multiHotLabel = getImgLabel( inferDir, i )
        #multiHotLabel = multiHotLabel.tolist()
        
##        # Skip images if needed.
##        if i.find( '_ni_' ) == -1:   continue
        
        ## Prediction from network.
        #img = cv2.imread( os.path.join( inferDir, 'images', i ) )
        #img1 = copy.deepcopy( img )
        #img2 = copy.deepcopy( img )
        #imgBatch = np.array( [ img ] )
        
        #inferLayerOut, inferPredLogits, inferPredResult, _, _ = \
                                            #tyClassifier.batchInference( imgBatch )
        
        #detectedBatchClassScores, _, detectedBatchClassNames, detectedBatchBboxes \
                                            #= nonMaxSuppression( inferPredResult )
        
        ## The output of the nonMaxSuppression is in the form of a batch.
        ## So extracting the contents of this batch since there is an output 
        ## of only one image in this batch.        
        #detectedBatchClassScores = detectedBatchClassScores[0]
        #detectedBatchClassNames = detectedBatchClassNames[0]
        #detectedBatchBboxes = detectedBatchBboxes[0]
                
##-------------------------------------------------------------------------------

        ## Draw the ground truth results now.
        #for l in labelDictList:
            #tlX, tlY, bboxW, bboxH = l['tlX'], l['tlY'], l['bboxW'], l['bboxH']
            #posX, posY = l['posX'], l['posY']
            #trueName = l['className']

            ## Only draw the bounding boxes for the non-rbc entities.
            #if trueName != '_' and trueName != 'Infected':
                #cv2.rectangle( img1, (tlX, tlY), (tlX + bboxW, tlY + bboxH), (0,255,0), 2 )
                #cv2.putText( img1, trueName[0], (posX, posY), \
                             #cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2, cv2.LINE_AA )
            #elif trueName == '_':       # Mark true normal rbc cells in green.
                #cv2.rectangle( img1, (tlX, tlY), (tlX + bboxW, tlY + bboxH), (0,255,0), 2 )
                #cv2.putText( img1, 'n', (posX, posY), \
                             #cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2, cv2.LINE_AA )
            #elif trueName == 'Infected':    # Mark true infected rbc cells in green.
                #cv2.rectangle( img1, (tlX, tlY), (tlX + bboxW, tlY + bboxH), (0,255,0), 2 )
                #cv2.putText( img1, 'i', (posX, posY), \
                             #cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2, cv2.LINE_AA )

##-------------------------------------------------------------------------------

        ## Draw the detected results now.
        #for pdx, p in enumerate( detectedBatchClassNames ):
            #x, y, w, h = detectedBatchBboxes[ pdx ].tolist()

            ## Only draw the bounding boxes for the non-rbc entities.
            #if p != '_' and p != 'Infected':
                #cv2.rectangle( img1, (x, y), (x+w, y+h), (0,0,255), 2 )
                #cv2.putText( img1, p[0], (x+5, y+15), \
                             #cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2, cv2.LINE_AA )
            #elif p == '_':      # Mark predicted normal rbc cells in blue.
                #cv2.rectangle( img1, (x, y), (x+w, y+h), (255,0,0), 2 )
                #cv2.putText( img1, 'n', (x+5, y+15), \
                             #cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2, cv2.LINE_AA )
            #elif p == 'Infected':   # Mark predicted infected rbc cells in red.
                #cv2.rectangle( img1, (x, y), (x+w, y+h), (0,0,255), 2 )
                #cv2.putText( img1, 'i', (x+5, y+15), \
                             #cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2, cv2.LINE_AA )

            #score = detectedBatchClassScores[ pdx ]
            #print( p, score )

##-------------------------------------------------------------------------------

        ## Now we also explicitly calculate the precision and recall beteween
        ## the infected and normal rbc classes, because that is what shows the
        ## reliability of the system on the malaria detection.
        
        #for pdx, p in enumerate( detectedBatchClassNames ):
            #if p != '_' and p != 'Infected':     continue    # Skip non-rbc classes.

            #x, y, w, h = detectedBatchBboxes[ pdx ].tolist()
            
            ## Now trying to find which true bbox will have the max iou with this 
            ## predicted bbox, and that corresponding className is termed bestMatchBboxClassName.
            #maxIou, bestMatchBboxClassName = iouThresh, None
            #for l in labelDictList:
                #trueName = l['className']
                #if trueName != '_' and trueName != 'Infected':   continue # Skip non-rbc classes.
                
                #tlX, tlY, bboxW, bboxH = l['tlX'], l['tlY'], l['bboxW'], l['bboxH']
                
                #iou, _, _ = findIOU( [ x, y, w, h ], [ tlX, tlY, bboxW, bboxH ] )
                #if iou >= maxIou:   maxIou, bestMatchBboxClassName = iou, trueName
                    
            ## Positive means INFECTED.
            ## Negative means NORMAL.
            #if bestMatchBboxClassName == 'Infected' and p == 'Infected':  rbcTP += 1
            #elif bestMatchBboxClassName == 'Infected' and p == '_':  rbcFN += 1
            #elif bestMatchBboxClassName == '_' and p == 'Infected':  rbcFP += 1
            #elif bestMatchBboxClassName == '_' and p == '_':  rbcTN += 1
            
            ## If no proper matches are found then the bestMatchBboxClassName will 
            ## remain as None. So just continue in that case.
            ## Otherwise see if the bestMatchBboxClassName is same as the predicted 
            ## name, which will make it a true positive. 
            ## Else a false positive or false negative.
            #else:   continue
        
        #print( '{}/{}'.format( idx+1, nImgs ) )
        
##-------------------------------------------------------------------------------
                
        #print( '\nTime taken: {}'.format( prettyTime( time.time() - startTime ) ) )
        #cv2.imwrite( os.path.join( inferDir, 'saved', 'prediction_' + i ), img1 )

        #cv2.imshow( 'Image', img1 )
        #cv2.imshow( 'Original Image', img2 )
        #print( i )
        #key = cv2.waitKey(1)
        #if key & 0xFF == 27:    break    # break with esc key.
    
    #cv2.destroyAllWindows()

##-------------------------------------------------------------------------------

    ## Calculating the precision, recall (or sensitivity) and specificity of 
    ## the rbc cells. This is calculated over the entire dataset.
    #rbcPrecision = rbcTP / ( rbcTP + rbcFP )
    #rbcRecall = rbcTP / ( rbcTP + rbcFN )
    #rbcSpecificity = rbcTN / ( rbcTN + rbcFP )
    #rbcF1score = 2 * ( rbcPrecision * rbcRecall ) / ( rbcPrecision + rbcRecall )
                    
    #print( '\nRBC Precision: {:0.3f}, RBC Recall: {:0.3f}, RBC Specificity: {:0.3f}, ' \
           #'RBC F1score: {:0.3f}'.format( rbcPrecision, rbcRecall, rbcSpecificity, rbcF1score ) )





##===============================================================================
    

