# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Data association class with single nearest neighbor association and gating based on Mahalanobis distance
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

# imports
import numpy as np
from scipy.stats.distributions import chi2

# add project directory to python path to enable relative imports
import os
import sys
from student.filter import Filter

from student.measurements import Measurement, Sensor
from student.trackmanagement import Track
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

import misc.params as params 

class Association:
    '''Data association class with single nearest neighbor association and gating based on Mahalanobis distance'''
    def __init__(self):
        self.association_matrix = np.matrix([])
        self.unassigned_tracks = []
        self.unassigned_meas = []
        
    def associate(self, track_list, meas_list, KF):
             
        ############
        # Step 3: association:
        # - replace association_matrix with the actual association matrix based on Mahalanobis distance (see below) for all tracks and all measurements
        # - update list of unassigned measurements and unassigned tracks
        ############
        N = len(track_list)
        M = len(meas_list)
        self.association_matrix = np.ones((N, M)) * np.inf
        for i in range(N):
            track:Track = track_list[i]
            for j in range(M):
                meas:Measurement = meas_list[j]
                dist = self.MHD(track, meas, KF)
                if self.gating(dist, meas.sensor):
                    self.association_matrix[i,j] = dist

        self.unassigned_tracks = list(range(N))
        self.unassigned_meas = list(range(M))
        

        
        # the following only works for at most one track and one measurement
        # self.unassigned_tracks = [] # reset lists
        # self.unassigned_meas = []
        
        # if len(meas_list) > 0:
        #     self.unassigned_meas = [0]
        # if len(track_list) > 0:
        #     self.unassigned_tracks = [0]
        # if len(meas_list) > 0 and len(track_list) > 0: 
        #     self.association_matrix = np.matrix([[0]])
        
        ############
        # END student code
        ############ 
                
    def get_closest_track_and_meas(self):
        ############
        # Step 3: find closest track and measurement:
        # - find minimum entry in association matrix
        arr = self.association_matrix
        ind = np.unravel_index(np.argmin(arr, axis=None), arr.shape)
        if np.min(arr) == np.inf:
            return np.nan, np.nan
        arr = np.delete(arr, ind[0], 0)
        arr = np.delete(arr, ind[1], 1)
        self.association_matrix = arr

        track = self.unassigned_tracks[ind[0]]
        meas = self.unassigned_meas[ind[1]]

        self.unassigned_tracks.remove(track)
        self.unassigned_meas.remove(meas)

        return track, meas

        # - delete row and column
        # - remove corresponding track and measurement from unassigned_tracks and unassigned_meas
        # - return this track and measurement
        ############

        # # the following only works for at most one track and one measurement
        # update_track = 0
        # update_meas = 0
        
        # # remove from list
        # self.unassigned_tracks.remove(update_track) 
        # self.unassigned_meas.remove(update_meas)
        # self.association_matrix = np.matrix([])
            
        # ############
        # # END student code
        # ############ 
        # return update_track, update_meas     

    def gating(self, MHD, sensor:Sensor): 
        ############
        # Step 3: return True if measurement lies inside gate, otherwise False
        ############
        if MHD < chi2.ppf(params.gating_threshold, df=sensor.dim_meas):
            return True
        return False
        
        ############
        # END student code
        ############ 
        
    def MHD(self, track:Track, meas:Measurement, KF:Filter):
        ############
        # Step 3: calculate and return Mahalanobis distance
        ############
        sensor:Sensor = meas.sensor
        gamma = meas.z - sensor.get_hx(track.x)
        S = KF.S(track, meas, sensor.get_H(track.x))
        return gamma.T * np.linalg.inv(S) * gamma
        
        ############
        # END student code
        ############ 
    
    def associate_and_update(self, manager, meas_list, KF):
        # associate measurements and tracks
        
        self.associate(manager.track_list, meas_list, KF)
        print("Association matrix shape, (num track, num meas):", self.association_matrix.shape)

        # update associated tracks with measurements
        while self.association_matrix.shape[0]>0 and self.association_matrix.shape[1]>0:
            
            # search for next association between a track and a measurement
            ind_track, ind_meas = self.get_closest_track_and_meas()
            if np.isnan(ind_track):
                print('---no more associations---')
                break
            track:Track = manager.track_list[ind_track]
            
            # check visibility, only update tracks in fov    
            if not meas_list[0].sensor.in_fov(track.x):
                continue
            
            # Kalman update
            print('update track', track.id, 'with', meas_list[ind_meas].sensor.name, 'measurement', ind_meas)
            KF.update(track, meas_list[ind_meas])
            
            # update score and track state 
            manager.handle_updated_track(track)
            
            # save updated track
            manager.track_list[ind_track] = track
            
        # run track management 
        manager.manage_tracks(self.unassigned_tracks, self.unassigned_meas, meas_list)
        
        for track in manager.track_list:            
            print('track', track.id, 'score =', track.score)