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
[image5]: ./output_images/heatmaps.png
[image6]: ./output_images/label_heatmap.png
[image7]: ./output_images/bbox.png
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


###False positives and overlapping bounding boxes

I recorded the positions of positive detections in each frame of the video.  From the positive detections I updated a ringbuffer like heatmap (see [heatmap_buffer.py](./heatmap_buffer.py)) and then thresholded that map to identify vehicle positions, see [p5.py](./p5.py) line 396.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:
Here are six frames and their corresponding heatmaps:

![alt text][image5]

Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:

![alt text][image6]

Here the resulting bounding boxes are drawn onto the last frame in the series:

![alt text][image7]

The heatmap approach has two drawbacks: first the temporal filtering is bad if the object moves significantly during one heatmap integration period. Second, bounding boxes are rather unstable (wobbling) due to hard thresholding and the point. To circumvent these problems, I decided to use the heatmap just for initialization of new tracks and do the tracking in another module. In line 408 in [p5.py](./p5.py) new tracks are initialized from the integrated heatmap. In line 402 the association between valid tracks and the raw detections (not the integrated heatmap to avoid a filter cascade) is done. The association considers the intersection over union as an association criterion. The extensions of the detections turn out to be rather unstable, so the association is not very strict (see line 429). Finally in line 405 the track parameters are updated. If no measurements are available or associated, the track is predicted based on the estimated (image) velocity (see line 492 and 493).

###Discussion

The results obtained so far are quite promising and motivate to continue on this project. One drawback of the current system is the high amount of false positives. Using the integrated heatmap helped a lot here, but for other scenarios this might be a problem. One way to proceed could be to extend the number of training and testing examples for example based on the Udacity [dataset](https://github.com/udacity/self-driving-car/tree/master/annotations) or the TME Motorway [dataset](http://cmp.felk.cvut.cz/data/motorway/). With more training data a nonlinear SVM or even a neural network might perform better than the simple linear SVM applied here.
Another possible improvement would be to do the tracking in 3D world coordinates instead of the image 2D image plane. Object movement can be much better described in 3D than in the image.
Occlusions are also a remaining difficulty for the current tracker, this can be observed also in the [video](./p5.mp4). In principle one could reject the wrong measurements due to occlusions, but this would require more stable detections. Currently the detection dimensions change significantly from frame to frame. 
Finally runtime should be addressed. The current implementation takes about 1s per image which is far away from realtime, however I think this was not the main goal of this project. 
