from tkinter.filedialog import askopenfilename
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math

def get_otsu(hist,bins):
    hist_norm = hist.ravel()/hist.sum()
    Q = hist_norm.cumsum()
    fn_min = np.inf
    otsu_idx = 0
    print(len(bins))
    for i in range(0,len(bins-1)):
        p1,p2 = np.hsplit(hist_norm,[i]) # probabilities
        q1,q2 = Q[i],Q[len(bins)-1]-Q[i] # cum sum of classes
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
            otsu_idx = i
    return bins[otsu_idx]

def correct_rear_ys(is_rearing,yraw,majax,orient,bbh):
    if is_rearing:
        ypos = yraw + math.cos(orient)*0.5*majax
        return ypos
    else:
        ypos = yraw
        return ypos

def correct_rear_xs(is_rearing,xraw,majax,orient):
    if is_rearing:
        xpos = xraw + math.sin(orient)*0.5*majax
        return xpos
    else:
        return xraw

def correctRears(df):
    
    df['orientation-abs'] = df['orientation'].abs()
    or_hist, bin_edges = np.histogram(df['orientation-abs'].dropna().values, density=True,bins=100)
    orient_thresh = get_otsu(or_hist,bin_edges[:-1])
    
    df['ax-ratio'] = df['maj-ax']/df['min-ax'] 
    ar_hist, bin_edges = np.histogram(df['ax-ratio'].dropna().values, density=True,bins=100)
    ax_ratio_thresh = get_otsu(ar_hist,bin_edges[:-1])
    
    df['rear-automatic'] = ( (df['ax-ratio'] > ax_ratio_thresh) & (df['orientation-abs'] < orient_thresh))
    df['ypos'] = df.apply( lambda x: correct_rear_ys(x['rear-automatic'],x['yraw'],x['maj-ax'],x['orientation'],x['bb-height']) ,axis=1)
    df['xpos'] = df.apply( lambda x: correct_rear_xs(x['rear-automatic'],x['xraw'],x['maj-ax'],x['orientation']) ,axis=1)
    
    return df
    
# demo
if __name__ == '__main__':
    
    df = pd.read_csv(askopenfilename())
    print(df.head())
    
    df = df.iloc[:200,:]
    out_df = correctRears(df)
    fig,ax = plt.subplots(1,1)
    ax.plot(df['xraw'],df['yraw'],color='k')
    ax.plot(df['xpos'],df['ypos'],color='dimgray',linestyle='--')
    ax.set_ylim(1000,0)
    plt.show()
    