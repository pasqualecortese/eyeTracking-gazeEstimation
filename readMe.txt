Eye tracking and gaze estimation.
Pasquale Enrico Cortese 903147.
Image analysis and computer vision project.

The main scripts are:
- 'gazeTracker_cv.m' is the calibration version of the program; 
    it is used to record the datasets stored in the folder 'data'.
- 'gazeTracker_test.m' is the test version of the program.

In the folder 'data' the datasets are stored, together with 
"regression_model.m" which is the script to create the regression model
that will be loaded in the main scripts ('regressionModel.mat').
In the folder 'functions' it's possibile to find all the function 
needed to run the main scripts.
In the folder 'video' the video recordings from the webcam are stored.

INSTRUCTION FOR THE USER
Stay with your eyes from a distance of 50 cm from the computer,
the tip of the nose at the same height of the camera.
In detection phase, it's required for the head to be 
aligned with the camera.
It's first needed to register a dataset, so run "gazeTracker_cv.m".
When enough data is collected, run "regression_model.m" 
inside the "data" folder.
At this point, it's possible to test the program, 
running "gazeTracker_test.m"

