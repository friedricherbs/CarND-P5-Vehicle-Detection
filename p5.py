# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 20:19:50 2017

@author: Friedrich Kenda-Erbs
"""

import numpy as np
import cv2
from scipy.ndimage.measurements import label
from skimage.feature import hog
from moviepy.editor import VideoFileClip
import pickle

# Import own classes
from vehicle import Vehicle
from heatmap_buffer import HeatmapBuffer

# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), 
                                  transform_sqrt=True, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, 
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), 
                       transform_sqrt=True, 
                       visualise=vis, feature_vector=feature_vec)
        return features

# Define a function to compute binned color features  
def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel() 
    # Return the feature vector
    return features

# Define a function to compute color histogram features 
def color_hist(img, nbins=32):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

# Create classifier hypotheses on multiple scales
def find_cars_multi_scale(img, svc, color_space='RGB'):
    
    # Describe scales and yImg ROIs for classifier search
    box_data =  [(1.0,  [400, 528]), 
                 (1.5,  [400, 656]),
                 (2.0,  [528, 656])]
    
    window_list = []
    score_list  = []
    for scale, y_lims in box_data:
        # Find detection for this predefined scale and ROI
        windows, scores = find_cars(img, scale, svc, ylims=y_lims, color_space=color_space)
        window_list.append(windows)
        score_list.append(scores)
        
    # flatten lists
    flatten_windows = [y for x in window_list for y in x]
    flatten_scores  = [y for x in score_list for y in x]
    
    return flatten_windows, flatten_scores

# Detect vehicles for at given scale and ROI  
def find_cars(img, scale, svc, ylims=[None, None], color_space='RGB'):
          
    # Set ROI params              
    if ylims[0] == None:
        ystart = 0
    else:
        ystart = ylims[0]
        
    if ylims[1] == None:
        ystop = img.shape[0]
    else:
        ystop = ylims[1]
        
    img_tosearch = img[ystart:ystop,:,:]
    
    # Apply color conversion if other than 'RGB'
    if color_space != 'RGB':
        if color_space == 'HSV':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2YCrCb)
    else: ctrans_tosearch = np.copy(img)   
    
    # Rescale whole image to calculate hog features at requested scale. This is more efficient than rescaling the detections
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        
    # Set color channels
    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]
    
    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - 1

    # Use standard window size 64x64
    window = 64
    nblocks_per_window = (window // pix_per_cell) - 1
    cells_per_step = 2 # Instead of overlap, define how many cells to step
    
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step
              
    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
    
    # Create an empty list to receive positive detection windows
    on_windows = []  
    # Create an empty list for detection scores
    scores = []
    
    for xb in range(nxsteps):
        for yb in range(nysteps):
            
            # Step cells 
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_features = np.hstack((hog_feat1,hog_feat2,hog_feat3))
            
            # Set top left position
            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell
            
            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
            
            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)
            
            # Scale extracted features to be fed to classifier
            test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
            
            # Predict using your classifier and get confidence
            score = svc.decision_function(test_features)
            # If confidence is sufficient store box
            if score > min_classifier_score:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw  = np.int(window*scale)
                on_windows.append(((xbox_left, ytop_draw+ystart), (xbox_left+win_draw, ytop_draw+win_draw+ystart)))
                scores.append(score)
                
    return on_windows, scores

# Add heat to heatmap based on valid detections depending on their detection scores
def add_heat(heatmap, bbox_list, scores):
    # Iterate through list of detections
    for box,score in zip(bbox_list,scores):
        # Add heat proportional to detection score
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += score

    # Return updated heatmap
    return heatmap

def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

# Define a function to draw tracked vehicles
def draw_vehicles(img, vehicles, color=(0, 0, 255), thick=6):
    # Iterate through all tracked vehicles
    for v in vehicles:
        # Draw a rectangle for each track
        xleft_v    = np.int(v.fit_centroid_x - 0.5*v.fit_width)
        xright_v   = np.int(v.fit_centroid_x + 0.5*v.fit_width)
                
        ytop_v     = np.int(v.fit_centroid_y - 0.5*v.fit_height)
        ybot_v     = np.int(v.fit_centroid_y + 0.5*v.fit_height)
        cv2.rectangle(img, (xleft_v,ytop_v), (xright_v,ybot_v, ), color, thick)
        
        # Show prediction age for debugging purposes
        cv2.putText(img, '{}'.format(v.age_predicted), (int(v.fit_centroid_x), int(v.fit_centroid_y)), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 2)
    # Return the image with boxes drawn
    return img

# Predict all vehicles in this frame and cleanup orphan tracks
def predict_vehicles(vehicle_list):
    # Reset internal track variables for this frame
    for v in vehicle_list:
        v.reset_frame()
        
    # Remove orphan tracks
    vehicle_list = [v for v in vehicle_list if v.age_predicted<max_age_predicted]
    
    return vehicle_list
            
# Associate detections to already existing vehicle tracks        
def assoc_vehicles(hot_windows):
    # Iterate through all detected cars
    for det_idx, det in enumerate(hot_windows):
        # Calc detection bounding box
        xleft  = det[0][0]
        ytop   = det[0][1]
        xright = det[1][0]
        ybot   = det[1][1]
        
        # Search over all vehicles
        best_iou = -1.0
        best_track_idx = -1
        for track_idx, v in enumerate(vehicle_list):
            
            if v.age_predicted < max_age_predicted:
                # Calc track bounding box
                centerx_v  = v.fit_centroid_x
                xleft_v    = np.int(centerx_v - 0.5*v.fit_width)
                xright_v   = np.int(centerx_v + 0.5*v.fit_width)
                
                centery_v  = v.fit_centroid_y
                ytop_v     = np.int(centery_v - 0.5*v.fit_height)
                ybot_v     = np.int(centery_v + 0.5*v.fit_height)
                
                # Calc intersection over union between track and detection:
                area_det = (xright-xleft+1)*(ybot-ytop+1)
                area_v   = (xright_v-xleft_v+1)*(ybot_v-ytop_v+1)
                intersection = max(0, min(xright, xright_v) - max(xleft, xleft_v)) *  max(0, min(ybot, ybot_v) - max(ytop, ytop_v))
                union    = area_det+area_v-intersection
                iou = intersection/union
                
                if (iou > min_iou) and (iou > best_iou):
                    best_iou       = iou
                    best_track_idx = track_idx
         
        # Assign detection to best matching track
        if best_track_idx >= 0:
            vehicle_list[best_track_idx].allx.extend(np.arange(xleft, xright+1))
            vehicle_list[best_track_idx].ally.extend(np.arange(ytop, ybot+1))
             
# Init new vehicle tracks if overlap between tracks and integrated heatmap is too small
def init_new_vehicles(labels):
    # Iterate through labels in integrated heatmap
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        xleft  = np.min(nonzerox)
        ytop   = np.min(nonzeroy)
        xright = np.max(nonzerox)
        ybot   = np.max(nonzeroy)
        
        # Search over all vehicles for best overlap to associate
        max_iou = -1.0
        for v in vehicle_list:
            
            if v.age_predicted < max_age_predicted:
                centerx_v  = v.fit_centroid_x
                xleft_v    = np.int(centerx_v - 0.5*v.fit_width)
                xright_v   = np.int(centerx_v + 0.5*v.fit_width)
                
                centery_v  = v.fit_centroid_y
                ytop_v     = np.int(centery_v - 0.5*v.fit_height)
                ybot_v     = np.int(centery_v + 0.5*v.fit_height)
                
                # Calc intersection over union:
                area_det = (xright-xleft+1)*(ybot-ytop+1)
                area_v   = (xright_v-xleft_v+1)*(ybot_v-ytop_v+1)
                intersection = max(0, min(xright, xright_v) - max(xleft, xleft_v)) *  max(0, min(ybot, ybot_v) - max(ytop, ytop_v))
                union    = area_det+area_v-intersection
                iou = intersection/union
                
                if iou > max_iou:
                    max_iou = iou
                    
        # If overlap is small -> init new track
        if max_iou < min_iou:
            new_vehicle = Vehicle()
            new_vehicle.centroid_x = 0.5*(xleft+xright)
            new_vehicle.centroid_y = 0.5*(ytop+ybot)
            new_vehicle.width      = xright-xleft 
            new_vehicle.height     = ybot-ytop
            new_vehicle.fit_centroid_x = new_vehicle.centroid_x
            new_vehicle.fit_centroid_y = new_vehicle.centroid_y
            new_vehicle.fit_width      = new_vehicle.width
            new_vehicle.fit_height     = new_vehicle.height
            new_vehicle.allx.extend(np.arange(xleft, xright+1))
            new_vehicle.ally.extend(np.arange(ytop, ybot+1))
            new_vehicle.detected      = True
            new_vehicle.age_predicted = 0
            vehicle_list.append(new_vehicle)
        
        
# Update vehicle track parameters                    
def update_vehicle_params():
    # Search over all vehicles
    for v in vehicle_list:
        if (v.age_predicted < max_age_predicted) and (len(v.allx) > 0) and (len(v.ally)) > 0:         
            # In this case we have a valid track and some measurements were associated so we can update the track parameters
            
            # Get last centroid position for velocity calculation
            last_centroid_x = v.centroid_x
            last_centroid_y = v.centroid_y
            
            # Update centroid position
            v.centroid_x = np.mean(v.allx)
            v.centroid_y = np.mean(v.ally)
            
            # Calc image displacement to last frame
            v.vx         = v.centroid_x-last_centroid_x
            v.vy         = v.centroid_y-last_centroid_y
                       
            # Calc dimensions
            v.height = np.amax(v.ally) - np.amin(v.ally)
            v.width  = np.amax(v.allx) - np.amin(v.allx)
            
            # Update filtered centroid position and dimensions
            v.fit_centroid_x = low_pass_constant*v.fit_centroid_x + (1.0-low_pass_constant)*v.centroid_x
            v.fit_centroid_y = low_pass_constant*v.fit_centroid_y + (1.0-low_pass_constant)*v.centroid_y
            v.fit_width      = low_pass_constant*v.fit_width      + (1.0-low_pass_constant)*v.width
            v.fit_height     = low_pass_constant*v.fit_height     + (1.0-low_pass_constant)*v.height
                                                                    
            if (v.fit_vx is not None) and (v.fit_vy is not None):
                # Update filtered velocity
                v.fit_vx     = low_pass_constant*v.fit_vx + (1.0-low_pass_constant)*v.vx
                v.fit_vy     = low_pass_constant*v.fit_vy + (1.0-low_pass_constant)*v.vy
            else:
                # If not set already, set filtered velocity
                v.fit_vx     = v.vx
                v.fit_vy     = v.vy
                  
            # Reset prediction age                                                  
            v.age_predicted = 0
            v.detected = True
        elif (v.age_predicted < max_age_predicted):
            if (v.fit_vx is not None) and (v.fit_vy is not None):
                # No measurements were associated in this cycle, just predict the track
                # No prediction if no velocity estimate available (e.g. track loss after 1 frame)
                v.centroid_x     += v.fit_vx
                v.centroid_y     += v.fit_vy
                v.fit_centroid_x += v.fit_vx
                v.fit_centroid_y += v.fit_vy
    

# Process an input image and yield output image showing valid vehicle tracks
def process_image(image):
    global frame_cnt
    global heatmap_buffer
    global vehicle_list
    
    # For drawing
    draw_image = np.copy(image)
    
    # Rescale image if input image are jpeg in the range 0 ... 255
    image = image.astype(np.float32)/255
           
    # Predict vehicle track list, reset orphan tracks
    vehicle_list = predict_vehicles(vehicle_list)  

    # Find valid vehicle detections
    hot_windows, scores = find_cars_multi_scale(image, svc, color_space=color_space)  

    # Update integrated heatmap
    current_heatmap = np.zeros((720,1280), dtype=np.float32)
    current_heatmap = add_heat(current_heatmap, hot_windows, scores)
    heatmap_buffer.add_heatmap(current_heatmap)                  
    
    # Apply threshold to heatmap to obtain valid objects
    integrated_heatmap = heatmap_buffer.apply_threshold(heatmap_threshold)
    
    # Find connected components describing vehicles
    labels = label(integrated_heatmap)
    
    # Associate valid detections to vehicle tracks
    assoc_vehicles(hot_windows)
    
    # Update tracks parameters
    update_vehicle_params()
    
    # Initialize new vehicle tracks if no overlap between heatmap vehicles and 
    init_new_vehicles(labels)

    # Draw vehicle tracks on image
    window_img = draw_vehicles(draw_image, vehicle_list)      
    
    frame_cnt += 1
    
    return window_img

# Parameter definitions
color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
spatial_size = (32, 32) # Spatial binning dimensions
hist_bins = 32    # Number of histogram bins
y_start_stop = [400, 656] # Min and max in y to search in slide_window()
heatmap_buffer_length = 5 # Ring buffer length of integrated heatmap
heatmap_threshold = 7 # Heatmap threshold. All labels above will initialize new tracks
min_classifier_score = 0.0 # Minimum classifier confidence to accept detection
max_age_predicted = 20 # Predict a valid track up to N frames without any detections 
min_iou = 0.01 # Minimum intersection over union value for measurement to track association and new track initialization
low_pass_constant = 0.8 # Low pass filter constant for shape and image velocity filtering

# Load classifier data generated in train.py
features  = pickle.load( open( "classifier.p", "rb" ) )
X_scaler  = features['X_scaler']
svc       = features['svc']

# New heatmap to filter detections
heatmap_buffer = HeatmapBuffer(heatmap_buffer_length)

# Vehicle track list
vehicle_list = []

# Video processing
frame_cnt = 1
print('Processing video ...')
clip = VideoFileClip('project_video.mp4')
vid_clip = clip.fl_image(process_image)
out_file = 'p5.mp4'
vid_clip.write_videofile(out_file, audio=False)