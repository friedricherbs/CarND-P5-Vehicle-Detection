# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 07:13:59 2017

@author: Friedrich Kenda-Erbs
"""

# Define a class to receive the characteristics of a tracked vehicle
class Vehicle():
    def __init__(self):
        # was the vehicle detected in the last iteration?
        self.detected = False  
        # current centroid x position
        self.centroid_x = -1.0
        # current centroid y position
        self.centroid_y = -1.0
        # current width estimate
        self.width = -1.0
        # current height estimate
        self.height = -1.0
        # current x velocity estimate
        self.vx = None
        # current < velocity estimate
        self.vy = None
        # current height estimate
        self.height = -1.0
        # averaged centroid x position over last cycles
        self.fit_centroid_x = -1.0
        # averaged centroid y position over last cycles
        self.fit_centroid_y = -1.0
        #x averaged width estimation
        self.fit_width = -1.0
        # averaged height estimation
        self.fit_height = -1.0
        # averaged x velocity estimate
        self.fit_vx = None
        # averaged y velocity estimate
        self.fit_vy = None
        #x values for detected pixels
        self.allx = []  
        #y values for detected pixels
        self.ally = []
        # number of frames vehicle is predicted
        self.age_predicted = 10000
     
    # Prepare track for new frame
    def reset_frame(self):
        self.detected            = False
        self.age_predicted       = self.age_predicted + 1
        self.allx                = []
        self.ally                = []