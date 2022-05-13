# dmbs-tracking
Position tracking tools for videos with temporally-varying illumination

# Requirements
[Anaconda](https://www.anaconda.com/)

# Usage instructions
## 1. Create a new configured environment and activate it

In the /dmbs-tracking folder:
```
conda create --name dmbs_tracking_env --file requirements.txt
activate dmbs_tracking_env
```

## 2. Generate a background model from the video
This script generates a background model from a provided video and saves it as a .png file in the background models folder. The solution assumes the presence of a single moving foreground object (such as an animal) within an arena fixed with reference to the camera field of view for the entirety of the video. As such, it does not require any segment of the video to not have the foreground object. 

This script can be run from the command line:

```
python estimate_background.py [start time in seconds]
```

Where `[start time in seconds]` should be set to the point in the video at which background estimation should begin. Frames prior to this time will be ignored for background model generation. This can be useful if there are extreme differences between the first few seconds of a video and the rest of the video. For example, you can use this parameter to cut out early parts of a recording with experiment intervention, etc.

After running the script, a dialog window will pop up. Select video for which to generate a background model. Next, a single frame will be shown. Use the crosshairs to select the mouse and press enter when done.

The background estimator will take a few minutes to run. When finished, it will display a plot of the background models. Close the window to exit the script. Your images will be saved in the background models folder.

## 3. Run the tracker
This script tracks the position of a foreground object in a video with varying illumination conditions given the video the analyze, a background model, and an ROI within the frame. The x- and y- coordinates of the foreground object over time are saved to a .csv file in a folder selected by the user. The script transforms position from pixel space to real space and corrects for perspective distortion (given real dimensions and corners of a rectangular arena).

```
python track_dmbs_perspective.py --perspective
```
The script will then create dialog windows to prompt you to select (in order):
1. The video file (.avi or .mp4 currently supported)
2. The background model (.png currently supported)
3. The folder to save the tracking data 
4. The ROI in the frame to analyze (use crosshairs to select the ROI, then press ENTER followed by ESC)

The script will then start tracking the foreground object and create a progressbar in the commandline terminal.

When it is finished tracking, it will begin transforming from pixel space to real space. The script will prompt you to select the corners of the rectangular arena. You must follow the exact order of the corners. Then enter (in order) the actual dimensions of the rectangular arena (in mm).

# Implementation notes

## Background estimation
The solution is based on weighted frame summing but has two key modifications to avoid including foreground objects in the background. First, the algorithm maintains a slow-moving average frame, and ignores frames that are not significantly different from the moving average. Second, it masks parts of a given frame that are different from the moving average, and contributes only that masked frame to the weighted adder. These modifications improve on simple summing techniques by avoiding erroneously including the foreground object in the background model when it is stationary for a long time.

## DMBS Tracker
