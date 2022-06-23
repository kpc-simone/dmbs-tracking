# background_estimation.py

import cv2 as cv2
import numpy as np
from matplotlib import pyplot as plt
from tkinter.filedialog import askopenfilename
from skimage.filters import gaussian
from skimage.color import rgb2gray
from skimage.morphology import disk, binary_erosion, binary_opening, binary_closing, closing
from skimage.segmentation import flood_fill
from skimage.measure import label, regionprops, regionprops_table
from matplotlib.gridspec import GridSpec
import progressbar
from PIL import Image
import scipy.ndimage as nd
from scipy import stats
import pandas as pd
import math
import sys,os
import time
import csv

def estimate_background(vidcap,test_frame,pos_0=50,change_thresh=8000,alpha_ = 0.01,wf_max=0.01):
    # compute algorithm steps
    FPS = int(vidcap.get(cv2.CAP_PROP_FPS))
    prev_frame = np.float32(test_frame)
    
    wa = 0.01 * prev_frame
    wa_mov = prev_frame
    ones = np.ones_like(wa_mov)

    model_built = False

    ts = [0]
    ws = [0]
    cs = [0]
    dfms = [127]

    vidcap.set(cv2.CAP_PROP_POS_FRAMES,int(pos_0*FPS))
    f_start = vidcap.get(cv2.CAP_PROP_POS_FRAMES)
    with progressbar.ProgressBar( max_value = 251.0) as pbar:
        while vidcap.isOpened():
            success,frame = vidcap.read()

            if frame is None:
                break
            
            #frame = np.float32(getROI(frame,roi))
            change = np.absolute(frame - wa_mov)
            
            cv2.accumulateWeighted(frame,wa_mov,alpha = alpha_ )
            ret,thresh = cv2.threshold(change[:,:,2],50,255,cv2.THRESH_BINARY_INV)
            mask = np.uint8(thresh)
            
            ret,thresh_inv = cv2.threshold(change[:,:,2],50,255,cv2.THRESH_BINARY)
            mask_inv = np.uint8(thresh_inv)
            
            change_med = np.median(np.median(change[:,:,2],axis=0),axis=0)
            #change_med = 10
            change_rel = (change[:,:,2] - change_med ).sum().sum() / 255
            weighting_factor = wf_max / ( 1 + np.exp( - ( change_rel - change_thresh ) ) )
            
            cv2.accumulateWeighted(frame,wa,weighting_factor,mask)
            cv2.accumulateWeighted(ones*np.median(np.median(wa_mov,axis=0),axis=0)*1.1,wa,weighting_factor,mask_inv)
            
            diff = wa_mov - wa
            diff_metric = diff.mean().mean().mean()
            #if weighting_factor > 0.8*wf_max:
            #    print(diff_metric)
            
            ts.append(ts[-1] + 1/FPS)
            ws.append(weighting_factor)
            cs.append(change_rel)
            dfms.append(diff_metric)
            
            pbar.update(255.0 - diff_metric)
            if abs(diff_metric) < 5.0:
                bg_model = wa
                model_built = True
                
                f_built = vidcap.get(cv2.CAP_PROP_POS_FRAMES)
                f_used = f_built - f_start

                out_data = {
                    'time'          : ts,
                    'change'        : cs,
                    'weighting'     : ws,
                    'diff metric'   : dfms,
                }
                out_df = pd.DataFrame(data=out_data)
                
                return bg_model,out_df,f_used,wa_mov,change

def select_video():
    vidpath = askopenfilename(
        title='Select video for which to estimate background',
        filetypes=[('Video Files', '*.avi; *.MP4'), ('All Files', '*.*')]
        )
    vidname = vidpath.split('/')[-1][:-4]
    ext = vidpath.split('/')[-1][-4:]
    vidcap = cv2.VideoCapture(vidpath)
    
    return vidcap,vidname,ext
    
def generate_test_frame(vidcap,pos_0=30):  
    FPS = int(vidcap.get(cv2.CAP_PROP_FPS))
    vidcap.set(cv2.CAP_PROP_POS_FRAMES,int(pos_0*FPS))
    success,test_frame = vidcap.read()
    vidcap.set(cv2.CAP_PROP_POS_FRAMES,0)
    return test_frame

def get_change_thresh(test_frame):
    cv2.namedWindow("Change threshold selection", cv2.WINDOW_NORMAL) 
    box_s = cv2.selectROIs("Change threshold selection", test_frame, fromCenter=False)
    ((xmin,ymin,width,height),) = tuple(map(tuple, box_s))
    cv2.destroyAllWindows()
    return width*height

def selectROI(test_frame):
    cv2.namedWindow("ROI Selection", cv2.WINDOW_NORMAL) 
    box_s = cv2.selectROIs("ROI Selection", test_frame, fromCenter=False)
    ((xmin,ymin,width,height),) = tuple(map(tuple, box_s))
    roi = {
        'xmin'      : xmin,
        'ymin'      : ymin,
        'width'     : width,
        'height'    : height,
    }
    cv2.destroyAllWindows()
    return roi

def getROI(frame,roi):
    
    xmin = roi['xmin']
    ymin = roi['ymin']
    width = roi['width']
    height = roi['height']
    
    roi = frame[ymin:ymin+height,xmin:xmin+width]
    
    return roi
    
def save_data():
    pass

if __name__ == '__main__':
    pos_0 = int(sys.argv[1])

    vidcap,vidname,ext = select_video()
    test_frame = generate_test_frame(vidcap,pos_0=pos_0)
    change_thresh_ = get_change_thresh(test_frame)
    
    t_start = time.time()
    bg_model,out_df,f_used,wa_mov,change = estimate_background(vidcap,test_frame,change_thresh = change_thresh_,pos_0=pos_0)
    t_built = time.time()
    t_run = round(t_built-t_start,2)
    
    print('background model for {}{}:\nbuilt with {} frames in {} sec of real time'.format(vidname,ext,int(f_used),t_run))
    
    fig, axes = plt.subplots(1,3,squeeze=False)
    axes[0,0].imshow(cv2.convertScaleAbs(bg_model))
    axes[0,0].set_title('Thresholded averaging with masking')
    
    axes[0,1].imshow(cv2.convertScaleAbs(wa_mov))
    axes[0,1].set_title('Moving average only')
    
    axes[0,2].imshow(cv2.convertScaleAbs(change))
    axes[0,2].set_title('Change from moving average')

    for ax in axes.ravel():
        ax.set_axis_off()
    
    print(out_df.head())
    
    plt.show()
    outdir = 'out data/bg'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    cv2.imwrite(os.path.join(outdir,'{}_bg.png'.format(vidname)),bg_model)
    # cv2.imwrite(os.path.join(outdir,'{}_wamov.png'.format(vidname)),wa_mov)
    # cv2.imwrite(os.path.join(outdir,'{}_change.png'.format(vidname)),change)
    # out_df.to_csv(os.path.join(outdir,'{}.csv'.format(vidname)))
    # metadatafile = 'metadata.csv'
    # if not os.path.exists(os.path.join(outdir,metadatafile)):
        # row = ['video','frames used','run time']
        # with open(os.path.join(outdir,metadatafile), 'a', newline='') as mdFile:
            # writer = csv.writer(mdFile)
            # writer.writerow(row)
        # mdFile.close()
    # with open(os.path.join(outdir,metadatafile), 'a', newline='') as mdFile:
        # row = ['{}{}'.format(vidname,ext),int(f_used),t_run]
        # writer = csv.writer(mdFile)
        # writer.writerow(row)
    # mdFile.close()