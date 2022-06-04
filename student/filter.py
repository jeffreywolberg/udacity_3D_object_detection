# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Kalman filter class
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

# imports
import misc.params as params
import numpy as np

# add project directory to python path to enable relative imports
import os
import sys
from student.measurements import Measurement, Sensor

from student.trackmanagement import Track
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))


class Filter:
    '''Kalman filter class'''

    def __init__(self):
        self.dt = params.dt
        self.q = params.q  # process noise variable for Kalman filter Q

    def F(self):
        ############
        # Step 1: implement and return system matrix F
        ############
        return np.matrix([
        [1, 0, 0, self.dt, 0, 0], 
        [0, 1, 0, 0, self.dt, 0], 
        [0, 0, 1, 0, 0, self.dt], 
        [0, 0, 0, 1, 0, 0], 
        [0, 0, 0, 0, 1, 0], 
        [0, 0, 0, 0, 0, 1]])

        ############
        # END student code
        ############

    def Q(self):
        ############
        # Step 1: implement and return process noise covariance Q
        ############
        q = self.q
        dt = self.dt
        return np.matrix([
            [q/3*dt**3, 0, 0, q/2*dt**2, 0, 0],
            [0, q/3*dt**3, 0, 0, q/2*dt**2, 0],
            [0, 0, q/3*dt**3, 0, 0, q/2*dt**2],
            [q/2*dt**2, 0, 0, q * dt, 0, 0],
            [0, q/2*dt**2, 0, 0, q*dt, 0],
            [0, 0, q/2*dt**2, 0, 0, q*dt],
        ])
        ############
        # END student code
        ############

    def predict(self, track: Track):    
        ############
        # Step 1: predict state x and estimation error covariance P to next timestep, save x and P in track
        ############
        F = self.F()
        Q = self.Q()
        x = F * track.x
        P = F * track.P * F.T + Q

        track.set_x(x)
        track.set_P(P)
        ############
        # END student code
        ############

    def update(self, track:Track, meas:Measurement):
        ############
        # TODO Step 1: update state x and covariance P with associated measurement, save x and P in track
        ############
        sensor:Sensor = meas.sensor
        gamma = self.gamma(track, meas)
        H = sensor.get_H(track.x)
        S = self.S(track, meas, H)
        K = track.P * H.T * np.linalg.inv(S) # Kalman Gain
        x = track.x + K * gamma
        P = (np.identity(6) - K*H) * track.P

        track.set_x(x)
        track.set_P(P)
        ############
        # END student code
        ############
        track.update_attributes(meas)

    def gamma(self, track:Track, meas:Measurement):
        ############
        # Step 1: calculate and return residual gamma
        ############
        sensor:Sensor = meas.sensor
        return meas.z - sensor.get_hx(track.x)
        ############
        # END student code
        ############

    def S(self, track : Track, meas : Measurement, H):
        ############
        # Step 1: calculate and return covariance of residual S
        ############
        return H * track.P * H.T + meas.R
        ############
        # END student code
        ############
