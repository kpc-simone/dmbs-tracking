from tkinter.filedialog import askopenfilename
import pandas as pd
import sys,os

sys.path.append(os.path.join(os.path.dirname(__file__),'src'))
from transformation import *
from detect_rearing import *

if __name__ == '__main__':
    
    bg_full = cv2.imread(askopenfilename(
        title='Select background model image file',
        filetypes=[('Image Files', '*.png'), ('All Files', '*.*')]
        )
    )
    
    tracking_filepath = askopenfilename(
        title='Select tracking data file to correct',
        filetypes=[('CSV Files', '*.csv'), ('All Files', '*.*')]
    )

    tracking_basename = os.path.basename(tracking_filepath)[:-4]
    
    # get warping parameters from arena corners and dimensions in real space
    corners = selectArenaCorners(bg_full)
    saveCorners(tracking_basename,corners)
    rect = loadCorners(tracking_basename)
    
    xdim = int(input('enter actual width (mm): '))
    ydim = int(input('enter actual depth (mm): '))
    known_dims = (xdim,ydim)
    H = getTransformParams(rect,known_dims)
    
    # apply warping to tracking data in pixel space
    out_df = pd.read_csv(tracking_filepath)
    out_df = correctRears(out_df)           # catch rearing bouts
    xsc,ysc = correctAllPoints(out_df['xpos'],out_df['ypos'],H)
    
    out_df['xcorr'] = xsc
    out_df['ycorr'] = ysc
    print(out_df.head())
    
    out_dir = os.path.dirname(tracking_filepath)
    out_df.to_csv(os.path.join(out_dir,'{}-corrected.csv'.format(tracking_basename)))