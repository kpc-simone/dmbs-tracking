# histogram_play.py

# histogram_play.py

import cv2 as cv2
import numpy as np
from matplotlib import pyplot as plt
from tkinter.filedialog import askopenfilename,askdirectory
from skimage.filters import gaussian
from skimage.color import rgb2gray
#from skimage.morphology import disk, binary_erosion, binary_dilation, erosion, dilation
from skimage.morphology import disk, binary_closing
from skimage.measure import label, regionprops, regionprops_table
from skimage.segmentation import flood_fill
from matplotlib.gridspec import GridSpec
import scipy.ndimage as nd
from PIL import Image
import progressbar
import pandas as pd
import imageio
import sys,os
import time
import math

sys.path.append(os.path.join(os.path.dirname(__file__),'src'))
from transformation import *
from detect_rearing import *

def getROI(frame,roi):
    
    xmin = roi['xmin']
    ymin = roi['ymin']
    width = roi['width']
    height = roi['height']
    
    roi = frame[ymin:ymin+height,xmin:xmin+width]
    
    return roi
    
def resize(image):
    image = Image.fromarray(image,'RGB')
    return image.thumbnail((128,128),Image.ANTIALIAS)

def merge_labels(labels_image,labels_to_merge,label_after_merge):
    labels_map = np.arange(np.max(labels_image)+1)
    labels_map[labels_to_merge] = label_after_merge
    return labels_map[labels_image]

if __name__ == '__main__':
    args = sys.argv[1:]

if '--plot' in args:
    plot = True
    
    # initialize figure
    fig = plt.figure( figsize = (10,4) )
    gs = GridSpec(3,5,figure=fig)
    
    ax_bg = fig.add_subplot(gs[1,0])
    ax_bg.set_title('Median-scaled\nbackground')
    
    ax_diff_g = fig.add_subplot(gs[1,1])
    ax_diff_g.set_title('Difference frame')
    
    ax_thresh_g = fig.add_subplot(gs[1,2])
    ax_thresh_g.set_title('Thresholded')
    
    ax_infr = fig.add_subplot(gs[:,3:])
    ax_infr.set_title('Object position\n(Estimated)')
    
    for ax in (fig.axes):
        ax.set_axis_off()
    
else: 
    plot = False

print('select video file for tracking')
videofile = askopenfilename(
    title='Select video to analyze',
    filetypes=[('Video Files', '*.avi; *.MP4'), ('All Files', '*.*')]
    )

videofilename = os.path.basename(videofile)[:-4]
print(videofilename)

# initialize matrices
bg_full = cv2.imread(askopenfilename(
    title='select background model image file',
    filetypes=[('Image Files', '*.png'), ('All Files', '*.*')]
    )
)

outdir = askdirectory(title='Select output directory for tracking data')

vidcap = cv2.VideoCapture(videofile)
FPS = vidcap.get(cv2.CAP_PROP_FPS)

# video mid-point
vidcap.set(cv2.CAP_PROP_POS_FRAMES,int(cv2.CAP_PROP_FRAME_COUNT/2))         
success,frame0 = vidcap.read()

box_s = cv2.selectROIs("Select video frame ROI", frame0, fromCenter=False)
((xmin,ymin,width,height),) = tuple(map(tuple, box_s))
roi = {
    'xmin'      : xmin,
    'ymin'      : ymin,
    'width'     : width,
    'height'    : height,
}
cv2.destroyAllWindows()
vidcap.set(cv2.CAP_PROP_POS_FRAMES,0)

# initialize thresholding parameters
k = 11
#selem_e = disk(5)
selem_e = disk(3)
selem_d = disk(5)
inv_offset = 255
blocksize = 681
level = 51

bg_model = getROI(bg_full,roi)
bg_blur = cv2.blur(cv2.cvtColor(bg_model, cv2.COLOR_BGR2GRAY),(k,k)).astype(np.float32)
diff = np.zeros( (roi['height'],roi['width'],1) ).astype(np.float32)
at2 = np.zeros( (roi['height'],roi['width'],1) )
at = np.zeros( (roi['height'],roi['width'],1) )

# initialize tracking variables
x_old = 0
y_old = 0
xpos = 0
ypos = 0

# loop through video and display          
pos_0 = 0
pos_f = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)/FPS
t_idx = 0

t_tracking_start = time.time()

#MSEC set works, but does not provide correct .get
#FPS set works, and provides correct .get
vidcap.set(cv2.CAP_PROP_POS_FRAMES,int(pos_0*FPS)+1)
print(vidcap.get(cv2.CAP_PROP_POS_FRAMES)/FPS)

mj = []
mn = []
dr = []
xs = []
ys = []
cs = []
bbws = []
bbhs = []

bg_scaled = np.zeros_like(bg_blur)
mu_half_bg = np.median(np.median(bg_blur,axis=0),axis=0)
with progressbar.ProgressBar( max_value = ( pos_f - pos_0 ) ) as p_bar:
    while ( vidcap.get(cv2.CAP_PROP_POS_FRAMES)/FPS < pos_f ):
        #print(vidcap.get(cv2.CAP_PROP_POS_FRAMES)/FPS)
        #print(vidcap.get(cv2.CAP_PROP_POS_FRAMES)/N_frames)
        p_bar.update( vidcap.get(cv2.CAP_PROP_POS_FRAMES)/FPS - pos_0 )

        success,frame0 = vidcap.read()

        if frame0 is None:
            break

        frame0 = getROI(frame0,roi)
        frame_blur = cv2.blur(cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY),(k,k),(k,k)).astype(np.float64)
        
        # scale the background
        mu_half_frame = np.median(np.median(frame_blur,axis=0),axis=0)
        alpha_ = mu_half_frame / mu_half_bg #/ mu_half_bg
        bg_scaled = np.clip(alpha_*bg_blur,0,255)

        # compute difference between background and foreground
        diff1 = np.clip((bg_scaled - frame_blur) / bg_scaled,0,255)
        
        # maximize contrast
        diff = np.absolute( diff1  / diff1.max().max() * 255 )
        
        #mu_half_diff = np.median(np.median(diff,axis=0),axis=0)
        #diff = np.absolute( ( ( diff - mu_half_diff ) / ( diff.max().max() - mu_half_diff ) ) * 255 )
        #diff = subGradients(diff)
        
        hist = cv2.calcHist([np.uint8(diff)],[0],None,[256],[0,256])
        hist_norm = hist.ravel()/hist.sum()
        Q = hist_norm.cumsum()
        bins = np.arange(256)
        fn_min = np.inf
        otsu_level = 63
        for i in range(63,255):
            p1,p2 = np.hsplit(hist_norm,[i]) # probabilities
            q1,q2 = Q[i],Q[255]-Q[i] # cum sum of classes
            if q1 < 1.e-6 or q2 < 1.e-6:
                continue
            b1,b2 = np.hsplit(bins,[i]) # weights
            # finding means and variances
            m1,m2 = np.sum(p1*b1)/q1, np.sum(p2*b2)/q2
            v1,v2 = np.sum(((b1-m1)**2)*p1)/q1,np.sum(((b2-m2)**2)*p2)/q2
            # calculates the minimization function
            fn = v1*q1 + v2*q2
            if fn < fn_min:
                fn_min = fn
                otsu_level = i
        
        #ret,at = cv2.threshold(np.uint8(diff),otsu_level,255,cv2.THRESH_BINARY)        
        at = binary_closing(cv2.adaptiveThreshold(255-np.uint8(diff),255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,blocksize,otsu_level),selem_d)
        
        lbl_a = label(at,background=255)

        regions = [region for region in sorted(regionprops(lbl_a), key=lambda r: r.area, reverse=True)]
        confidence = 0
        for region in regions[1:]:
            area = region.area    
            if region.area > 250:
                yc, xc = region.centroid
                minr, minc, maxr, maxc = region.bbox
                bbh = maxr-minr
                bbw = maxc-minc
                
                orient = region.orientation
                majax = region.major_axis_length
                minax = region.minor_axis_length
                    
                ypos = yc
                xpos = xc
                
                confidence = region.area / np.sqrt( (xpos-x_old)**2 + (ypos-y_old)**2 ) / len(regions)
                log_confidence = np.log10(np.absolute(confidence))
                x_old = xpos
                y_old = ypos   
        
                bbhs.append(bbh)
                bbws.append(bbw)
                
                mj.append(majax)
                mn.append(minax)
                dr.append(orient)
                
                xs.append(xpos)
                ys.append(ypos)
                cs.append(confidence)
                break
            
            else:
                bbhs.append(np.nan)
                bbws.append(np.nan)
                
                mj.append(np.nan)
                mn.append(np.nan)
                dr.append(np.nan)
                
                xs.append(np.nan)
                ys.append(np.nan)
                cs.append(0)
                break
        if plot:
            
            if t_idx == 0:
                # plot the data
                plot_bg = ax_bg.imshow(cv2.convertScaleAbs(bg_model),cmap='gray')
                plot_diff_gray = ax_diff_g.imshow(np.uint8( diff ), cmap = 'gray')
                plot_thresh_gray = ax_thresh_g.imshow(at, cmap = 'gray')
                plot_infr = ax_infr.imshow(frame0)
                plot_pos, = ax_infr.plot(xpos,ypos,color='r',marker='.', markersize=10)
                
                plot_dict = {
                    'plot_bg'               : plot_bg,                   
                    'plot_diff_gray'        : plot_diff_gray,                     
                    'plot_thresh_gray'      : plot_thresh_gray,
                    'plot_infr'             : plot_infr,
                    'plot_pos'              : plot_pos,
                }
                
            else:
                
                plot_dict['plot_bg'].set_data(bg_scaled)
                plot_dict['plot_diff_gray'].set_data(diff)
                plot_dict['plot_thresh_gray'].set_data(at)
                
                plot_dict['plot_infr'].set_data(frame0)
                plot_dict['plot_pos'].set_data(xpos,ypos)
            
            fig.tight_layout()
            plt.draw()
            plt.pause(1e-17)
            frame1 = frame0
            frame2 = frame1
            t_idx += 1

t_tracking_end = time.time()

ts = np.linspace(pos_0,pos_f,len(xs))
out_data = {
    'time'              : ts,
    'xraw'              : xs+roi['xmin'],
    'yraw'              : ys+roi['ymin'],
    'bb-width'          : bbws,
    'bb-height'         : bbhs,
    'maj-ax'            : mj,
    'min-ax'            : mn,
    'orientation'       : dr,
    'confidence'        : cs,
}

print('Speed results:')
print('Tracking: {} for {} s of video'.format(t_tracking_end-t_tracking_start,pos_f-pos_0))

out_df = pd.DataFrame(data = out_data)
print(out_df.head())
print(out_df.tail())
out_df.to_csv(os.path.join(outdir,'{}-dmbs.csv'.format(videofilename)))