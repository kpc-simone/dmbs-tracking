# extract_ambient_lighting.py

from tkinter.filedialog import askopenfilename
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2

def getROI(frame,roi):
    
    xmin = roi['xmin']
    ymin = roi['ymin']
    width = roi['width']
    height = roi['height']
    
    roi = frame[ymin:ymin+height,xmin:xmin+width]
    
    return roi

ts = 400
t0 = 0

videofile = askopenfilename()
vidcap = cv2.VideoCapture(videofile)
FPS = vidcap.get(cv2.CAP_PROP_FPS)
vidcap.set(cv2.CAP_PROP_POS_FRAMES,int(ts * FPS))
success,frame = vidcap.read()

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
    
fig, ax = plt.subplots(1,1,squeeze=False,figsize=(8,3))
ax[0,0].plot(ts,intsy,color='k')
for spine in ['top','right']:
    ax[0,0].spines[spine].set_visible(False)
ax[0,0].set_xlabel('Time (s)')
ax[0,0].set_ylabel('Mirror mean pixel intensity')

fig.tight_layout()
plt.show()