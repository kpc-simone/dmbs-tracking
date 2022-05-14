# dmbs-tracking
Position tracking tools for behavior videos with temporally-varying illumination and/or angled views

![](https://github.com/kpc-simone/dmbs-tracking/blob/main/docs/dmbs_tracking_demo.gif)

# Requirements
- [Anaconda](https://www.anaconda.com/) 
- Open the Anaconda Navigator and launch Powershell
- use `cd` to navigate into the `dmbs-tracking` folder, then create a new configured environment:

	```
	conda create --name dmbs_tracking_env --file requirements.txt
	activate dmbs_tracking_env
	```

# Usage instructions

## 1. Generate a background model from the video

This script can be run from the command line:

```
python estimate_background.py [start time in seconds]
```

1. Where `[start time in seconds]` should be set to the point in the video at which background estimation should begin. 
2. After running the script, a dialog window will pop up. Select video for which to generate a background model. 
3. Next, a single frame will be shown. Use the crosshairs to select the mouse and press enter when done.
4. The background estimator will take a few minutes to run. When finished, it will display a plot of the background models. Close the window to exit the script. 
5. Your images will be saved in `out data/bg`

## 2. Run the tracker

```
python track_dmbs.py
```

The script will then create dialog windows to prompt you to select (in order):
1. The video file (.avi or .mp4 currently supported)
2. The background model (.png currently supported)
3. The folder to save the tracking data 
4. The ROI in the frame to analyze (use crosshairs to select the ROI, then press ENTER followed by ESC).

![](https://github.com/kpc-simone/dmbs-tracking/blob/main/docs/roi%20selection.png)

The script will then start tracking the foreground object and create a progressbar in the commandline terminal. When it is finished tracking, it will save the tracking data (as a .csv file) in the folder you specified.

## 3. Transform from pixel space to actual space

```
python correct_position.py --perspective
```
The script will create dialog windows to prompt you to select (in order):
1. The background model (.png currently supported)
2. The tracking data file to select(.avi or .mp4 currently supported)
3. The script will then show the background model image. Select the corners of the rectangular arena in the exact order requested (back left, back right, front right, front left). Then enter (in order) the actual dimensions of the rectangular arena (in mm).
The script will save the corrected tracking data (as a .csv file) in the same folder as the raw tracking data file selected. If `--perspective` is not given, no rearing detection/position correction will be done.

# Implementation notes

## Background estimation
This script generates a background model from a provided video and saves it as a .png file in the background models folder. The solution assumes the presence of a single moving foreground object (such as an animal) within an arena fixed with reference to the camera field of view for the entirety of the video. As such, it does not require any segment of the video to not have the foreground object. 

The solution is based on weighted frame summing but has two key modifications to avoid including foreground objects in the background. First, the algorithm maintains a slow-moving average frame, and ignores frames that are not significantly different from the moving average. Second, it masks parts of a given frame that are different from the moving average, and contributes only that masked frame to the weighted adder. These modifications improve on simple summing techniques by avoiding erroneously including the foreground object in the background model when it is stationary for a long time.

Frames prior to the given `start time` will be ignored for background model generation. This can be useful if there are extreme differences between the first few seconds of a video and the rest of the video. For example, you can use this parameter to cut out early parts of a recording with experiment intervention, etc.

## DMBS Tracker
This script tracks the position of a foreground object in a video with varying illumination conditions given the video the analyze, a background model, and an ROI within the frame. The x- and y- coordinates of the foreground object over time are saved to a .csv file in a folder selected by the user. The script transforms position from pixel space to real space and corrects for perspective distortion (given real dimensions and corners of a rectangular arena).

## Position correction and transformation
This script generates a homography to map pixel space to real space given real dimensions of a rectangular arena and its corners. For perspective-view videos, this recovers position as if the video had a top-down view

![](https://github.com/kpc-simone/dmbs-tracking/blob/main/docs/perspective_dist.png)

