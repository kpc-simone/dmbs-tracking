from tkinter.filedialog import askopenfilename
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys,os
import math
import csv
import cv2

arenaCorners = []
def getTransformParams(rect,known_dims):
    
    maxWidth, maxHeight = known_dims
    
    # construct the set of destination points to obtain top-down view)
    dst = np.float32([ 
        [int(-maxWidth/2), 0],
        [int(maxWidth/2) - 1, 0],
        [int(maxWidth/2) - 1, int(maxHeight) - 1],
        [int(-maxWidth/2) - 1, int(maxHeight) - 1]])
    
    # compute the perspective transform matrix
    H = cv2.findHomography(rect, dst,cv2.RANSAC,5.0)[0]
    return H
    
def correctPosition(point,H):
    point_corrected = np.zeros_like(point)
    point_corrected = np.squeeze(np.squeeze(cv2.perspectiveTransform(np.float32([point]), H),axis=0),axis=0)
    
    return point_corrected

def correctAllPoints(xs_to_correct,ys_to_correct,H):
    xs_corrected = []
    ys_corrected = []
    for x,y in zip(xs_to_correct,ys_to_correct):
        pc = correctPosition(np.array([[x,y]]),H)
        xs_corrected.append(pc[0]/1000)                     # convert from mm to m
        ys_corrected.append(pc[1]/1000)                     # convert from mm to m
        #print(pc)
        
    return xs_corrected,ys_corrected

def getImagePoints(img):
    # threshold and label image 
    img = img > 0.7
    #axes[0,2].imshow(im0,cmap='gray')                      # check thresholding
    #fig,ax = plt.subplots(1,1)
    #ax.imshow(img)
    #plt.show(block=False)

    lbl = label(img,background=1)
    n_regions = lbl.max()
    print('regions labelled: {}'.format(n_regions))
    
    #recreate model of dots and corrected points
    regions = [region for region in sorted(regionprops(lbl), key=lambda r: r.centroid, reverse=True)]
    points = np.array([[r.centroid[1],r.centroid[0]] for r in regions])
    
    return points

def selectCorner(event, x, y, flags, param):
    global arenaCorners
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x,",",y)
        
        arenaCorners.append([x,y])
        font = cv2.FONT_HERSHEY_SIMPLEX
        strXY = str(x)+", "+str(y)
    
def selectArenaCorners(frame):
    for corner in ['back left','back right','front right','front left']:
        cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
        cv2.imshow("frame",frame)
        print('identify {} corner'.format(corner))
        cv2.setMouseCallback("frame",selectCorner)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    global arenaCorners
    return arenaCorners
      
def saveCorners(recording,arenaCorners):
    metadatafile = '../out data/all-corners.csv'
    if not os.path.exists(os.path.join(os.path.dirname(__file__),metadatafile)):
        row = ['recording','back left','back right','front right','front left']
        with open(os.path.join(os.path.dirname(__file__),metadatafile), 'a', newline='') as mdFile:
            writer = csv.writer(mdFile)
            writer.writerow(row)
        mdFile.close()
        
    with open(os.path.join(os.path.dirname(__file__),metadatafile), 'a', newline='') as mdFile:
        row = [recording,arenaCorners[0],arenaCorners[1],arenaCorners[2],arenaCorners[3]]
        writer = csv.writer(mdFile)
        writer.writerow(row)
    mdFile.close()
    
def loadCorners(recording):
    cnrs_df = pd.read_csv(os.path.join(os.path.dirname(__file__),'../out data/all-corners.csv'))
    cnrs_row = cnrs_df[cnrs_df['recording'] == recording].iloc[0]
    print(cnrs_row)
    
    back_left = [int(e) for e in cnrs_row['back left'].strip( '][' ).split(',')]
    back_right = [int(e) for e in cnrs_row['back right'].strip( '][' ).split(',')]
    front_right = [int(e) for e in cnrs_row['front right'].strip( '][' ).split(',')]
    front_left = [int(e) for e in cnrs_row['front left'].strip( '][' ).split(',')]
    
    rect = np.float32([ 
        [back_left[0], back_left[1]],
        [back_right[0], back_right[1]],
        [front_right[0], front_right[1]],
        [front_left[0], front_left[1]],
        ])
    
    return rect
    