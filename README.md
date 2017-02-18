# Vehicle Detection Project
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_examples.png
[image2]: ./output_images/HOG_example.png
[image3]: ./output_images/sliding_windows.png
[image4]: ./output_images/example_detections.png
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

###Histogram of Oriented Gradients (HOG)

HOG features are extracted in [train.py](https://github.com/friedricherbs/CarND-P5-Vehicle-Detection/blob/master/train.py) in the method `get_hog_features()`. Basically this function uses the `scikit-image` hog feature implementation documented [here](http://scikit-image.org/docs/dev/api/skimage.feature.html?highlight=feature%20hog#skimage.feature.hog).

I started by reading in all the `vehicles` and `non-vehicles` images in [train.py](https://github.com/friedricherbs/CarND-P5-Vehicle-Detection/blob/master/train.py) in line 127-130.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the Y-channel of `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

I tried different colorspaces and HoG parameters. The final configuration was chosen according to the one which gave the best test accuracy from the classifier. Using all three HOG channels increases runtime and feature vector length significantly, however it has the biggest impact on the testing accuracy. The final parameter set includes 9 orientations (larger values did not improve the accuracy of the classifier much), 8 pixels_per_cell, (2, 2) cells_per_block and (2, 2) and `transform_sqrt` set to false (threw an error due to negative values). 

For classification I trained a linear SVM using the `scikit-learn fit()` function, see  [train.py](https://github.com/friedricherbs/CarND-P5-Vehicle-Detection/blob/master/train.py) in line 181. Using a non-linear SVM yielded clearly inferior test accuracies probably due to overfitting and took much longer so I did not follow this path.

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

A sliding window search is implemented in [p5.py](https://github.com/friedricherbs/CarND-P5-Vehicle-Detection/blob/master/p5.py) in the functions `find_cars_multi_scale()`starting in line 58 and `find_cars()` starting in line 80. Different scales were searched depending on the `yImg` coordinate: vehicles farther away appear smaller, closer vehicles appear wider. The different scales are summarized here:

| yImg          | Scale         | 
|:-------------:|:-------------:| 
| 400, 528      | 1             | 
| 400, 656      | 1.5           |
| 528, 656      | 2.0           |

The scale factor adapts the whole image size instead of rescaling the standard detection window size of 64x64 pixels. The step size was 2 cells times the scaling factor. An example image of the search images is shown here:


![alt text][image3]

Ultimately I searched on three scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided quite good results.  Here are some example images:

![alt text][image4]
---

### Video Implementation

Here's a [link to my video result](./p5.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
