# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 07:10:48 2017

@author: Friedrich Kenda-Erbs
"""
from collections import deque
import numpy as np

# Class for temporal integration of heatmap by means of a ringbuffer
class HeatmapBuffer:
    
    # Constructor
    def __init__(self, maxlen):
        # Setup new heatmap buffer
        self.data = deque(maxlen=maxlen)
        # Init data
        img = np.zeros((720,1280), dtype=np.float32)
        for i in range(maxlen):
            self.data.append(img)
     
    # Add new heatmap for one image to temporal ringbuffer
    def add_heatmap(self, heatmap):
        # Add heatmap to heatmap buffer
        self.data.append(heatmap)
     
    # Sum up all heatmaps in buffer and apply threshold
    def apply_threshold(self, threshold):
        sum_heatmap = sum(self.data) 
        # Zero out pixels below the threshold
        sum_heatmap[sum_heatmap <= threshold] = 0
        # Return thresholded map
        return sum_heatmap
            
            
        
        