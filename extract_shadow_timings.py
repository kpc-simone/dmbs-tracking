from tkinter.filedialog import askopenfilename
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from scipy import signal
import pandas as pd
import numpy as np
import sys,os
import cv2

### script parameters ###
shadow_duration_ = 8.       # the duration of the shadow; the script runs the video backwards
ignore_ = 60.               # the time interval at the beginning of the recording where no shadows are presented
threshold_factor_ = 3.      # the factor by which to scale the std dev during the baseline for intensity thresholding
###

def extract_shadow_timings(ts,intsy,FPS,recording,ignore,shadow_dur = 8.,threshold_factor = 3.):
    # get lowest value
    sos = signal.butter(6, FPS/2.1, 'low', fs=FPS, output='sos')
    intsy = signal.sosfilt(sos, intsy)
    
    # thresh = np.median(intsy) - 0.65 * ( np.median(intsy) - intsy.min() )
    thresh = np.median(intsy) - threshold_factor * np.std(intsy)
    
    shdf = pd.DataFrame(columns=['recording','shadowON-abs','shadowOFF-abs'])
    t = ts.max()
    idx = int(len(intsy)-1)
    
    fig,ax = plt.subplots(1,1,figsize=(8,3))
    while (t > ignore) & (idx > 1):
        val = intsy[idx]
        t = ts[idx]
        if val < thresh:
            ax.scatter(t,val,color='k')
            shdf = shdf.append({
                'recording'     : recording,
                'shadowON-abs'  : t - shadow_dur,
                'shadowOFF-abs'  : t,
                },ignore_index=True)
            idx -= int(FPS * shadow_dur)
        else:
            idx -= 1
    shdf = shdf.sort_values('shadowON-abs').reset_index(drop=True)
    shdf['trial'] = shdf.index + 1
    cols = shdf.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    shdf = shdf[cols]
    
    print(shdf.head(15))
    
    ax.plot(ts,intsy,color='dimgray')
    ax.axhline(thresh,color='k',linestyle='--')
    ax.text(0,thresh+1,'detection threshold')
    
    for r,row in shdf.iterrows():
        start = row['shadowON-abs']
        stop = row['shadowOFF-abs']
        ax.axvspan(start,stop,color='silver')
        ax.text(start,thresh,'shadow {}'.format(r+1),rotation=90)
    
    ax.set_ylabel('Mean ROI pixel intensity')
    ax.set_xlabel('Time (s)')
    
    for spine in ['top','right']:
        ax.spines[spine].set_visible(False)
    
    fig.tight_layout()
    plt.show()
    return shdf

def getROI(frame,roi):
    
    xmin = roi['xmin']
    ymin = roi['ymin']
    width = roi['width']
    height = roi['height']
    
    roi = frame[ymin:ymin+height,xmin:xmin+width]
    
    return roi

ts = 100
t0 = 0

videofile = askopenfilename()
vidcap = cv2.VideoCapture(videofile)
FPS = vidcap.get(cv2.CAP_PROP_FPS)
vidcap.set(cv2.CAP_PROP_POS_FRAMES,int(ts * FPS))
success,frame = vidcap.read()

cv2.namedWindow("Select mirror ROI",cv2.WINDOW_NORMAL)
box_s = cv2.selectROIs("Select mirror ROI", frame, fromCenter=False)
((xmin,ymin,width,height),) = tuple(map(tuple, box_s))
roi = {
    'xmin'      : xmin,
    'ymin'      : ymin,
    'width'     : width,
    'height'    : height,
}
cv2.destroyAllWindows()

N_FRAMES = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
intsy = np.zeros( (N_FRAMES-int(t0 * FPS),) )
ts = np.linspace(0,(N_FRAMES-int(t0 * FPS))/FPS,N_FRAMES-int(t0 * FPS))

t_idx = 0
vidcap.set(cv2.CAP_PROP_POS_FRAMES,int(t0 * FPS))
print('showing shadow state for {}'.format(videofile))
while vidcap.isOpened():
    success,frame = vidcap.read()

    if frame is None:
        break
    
    intsy_roi = getROI(frame,roi)
    intsy[t_idx] = intsy_roi.mean().mean()
    
    intsy_roi_prev = intsy[t_idx]
    t_idx += 1
    
shdf = extract_shadow_timings(ts,intsy,FPS,
                                recording = os.path.basename(videofile),
                                ignore = ignore_,
                                shadow_dur = shadow_duration_,
                                threshold_factor = threshold_factor_
                                )

videofilename = videofile.split('.')[0]
out_dir = os.path.dirname(videofilename)
shdf.to_csv(os.path.join(out_dir,'{}-shadowtimings.csv'.format(os.path.basename(videofilename))))