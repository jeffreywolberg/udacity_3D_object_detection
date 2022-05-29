# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Evaluate performance of object detection
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

# general package imports
import numpy as np
import matplotlib
matplotlib.use('wxagg') # change backend so that figure maximizing works on Mac as well     
import matplotlib.pyplot as plt

import torch
from shapely.geometry import Polygon
from operator import itemgetter

# add project directory to python path to enable relative imports
import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

# object detection tools and helper functions
import misc.objdet_tools as tools


# compute various performance measures to assess object detection
def measure_detection_performance(detections, labels, labels_valid, min_iou=0.5):
    
     # find best detection for each valid label 
    true_positives = 0 # no. of correctly detected objects
    center_devs = []
    ious = []
    for label, valid in zip(labels, labels_valid):
        matches_lab_det = []
        if valid: # exclude all labels from statistics which are not considered valid
            
            for det in detections:
                box = label.box
                rx = box.center_x + box.width/2
                lx = box.center_x - box.width/2
                ty = box.center_y - box.length/2
                by = box.center_y + box.length/2
                tz = box.center_z + box.height/2
                bz = box.center_z - box.height/2
                tr, tl, bl, br = (rx, ty), (lx, ty), (lx, by), (rx, by)
                gt_rect = Polygon([bl, br, tr, tl])
                # _, bev_x, bev_y, z, h, bev_w, bev_l, yaw = det
                _, x, y, z, w, l, h, _ = det
                det_lx = x
                det_rx = x + w
                det_ty = y
                det_by = y + l
                det_tr, det_tl, det_bl, det_br = (det_rx, det_ty), (det_lx, det_ty), (det_lx, det_by), (det_rx, det_by)
                det_rect = Polygon([det_bl, det_br, det_tr, det_tl])

                get_x_coord = lambda tup: tup[0]
                get_y_coord = lambda tup: tup[1]
                rightmost_left = max(bl, det_bl, key=get_x_coord)[0]
                leftmost_right = min(br, det_br, key=get_x_coord)[0]
                topmost_bottom = min(bl, det_bl, key=get_y_coord)[1]
                bottommost_top = max(tl, det_tl, key=get_y_coord)[1]
                iou_tr, iou_tl, iou_bl, iou_br = (leftmost_right, bottommost_top), (rightmost_left, bottommost_top), (rightmost_left, topmost_bottom), (leftmost_right, topmost_bottom)
                
                import cv2
                img = np.zeros((70, 70, 3))
                print([tr, tl, bl, br])
                print([det_tr, det_tl, det_bl, det_br])
                print([iou_tr, iou_tl, iou_bl, iou_br])
                img = cv2.polylines(img, [np.array([tr, tl, bl, br]).astype(np.int32)], isClosed=True, color=[0, 255, 0], thickness=2)
                img = cv2.polylines(img, [np.array([det_tr, det_tl, det_bl, det_br]).astype(np.int32)], isClosed=True, color=[255, 0, 0], thickness=2)
                color = [255, 255, 255] if leftmost_right-rightmost_left >= 0 and bottommost_top-topmost_bottom else [0, 0, 255]
                img = cv2.polylines(img, [np.array([iou_tr, iou_tl, iou_bl, iou_br]).astype(np.int32)], isClosed=True, color=color, thickness=1)

                if leftmost_right-rightmost_left >= 0 and bottommost_top-topmost_bottom:
                    intersection = abs(iou_tr[0]-iou_tl[0]) * abs(iou_tr[1] - iou_br[1])
                    union = abs(tr[0]-tl[0]) * abs(tr[1] - br[1]) + abs(det_tr[0]-det_tl[0]) * abs(det_tr[1] - det_br[1]) - intersection
                    calc_intersection = gt_rect.intersection(det_rect)
                    calc_union = gt_rect.union(det_rect)
                    print("My intersection: ", intersection)
                    print("Calculated intersection", calc_intersection.area)
                    print("My union:", union)
                    print("Calculated union:", calc_union.area)
        
                cv2.imshow("Im", img)
                cv2.waitKey(0)





            
            # compute intersection over union (iou) and distance between centers

            ####### ID_S4_EX1 START #######     
            #######
            print("student task ID_S4_EX1 ")

            ## step 1 : extract the four corners of the current label bounding-box
            
            ## step 2 : loop over all detected objects

                ## step 3 : extract the four corners of the current detection
                
                ## step 4 : computer the center distance between label and detection bounding-box in x, y, and z
                
                ## step 5 : compute the intersection over union (IOU) between label and detection bounding-box
                
                ## step 6 : if IOU exceeds min_iou threshold, store [iou,dist_x, dist_y, dist_z] in matches_lab_det and increase the TP count
                
            #######
            ####### ID_S4_EX1 END #######     
            
        # find best match and compute metrics
        if matches_lab_det:
            best_match = max(matches_lab_det,key=itemgetter(1)) # retrieve entry with max iou in case of multiple candidates   
            ious.append(best_match[0])
            center_devs.append(best_match[1:])


    ####### ID_S4_EX2 START #######     
    #######
    print("student task ID_S4_EX2")
    
    # compute positives and negatives for precision/recall
    
    ## step 1 : compute the total number of positives present in the scene
    all_positives = 0

    ## step 2 : compute the number of false negatives
    false_negatives = 0

    ## step 3 : compute the number of false positives
    false_positives = 0
    
    #######
    ####### ID_S4_EX2 END #######     
    
    pos_negs = [all_positives, true_positives, false_negatives, false_positives]
    det_performance = [ious, center_devs, pos_negs]
    
    return det_performance


# evaluate object detection performance based on all frames
def compute_performance_stats(det_performance_all):

    # extract elements
    ious = []
    center_devs = []
    pos_negs = []
    for item in det_performance_all:
        ious.append(item[0])
        center_devs.append(item[1])
        pos_negs.append(item[2])
    
    ####### ID_S4_EX3 START #######     
    #######    
    print('student task ID_S4_EX3')

    ## step 1 : extract the total number of positives, true positives, false negatives and false positives
    
    ## step 2 : compute precision
    precision = 0.0

    ## step 3 : compute recall 
    recall = 0.0

    #######    
    ####### ID_S4_EX3 END #######     
    print('precision = ' + str(precision) + ", recall = " + str(recall))   

    # serialize intersection-over-union and deviations in x,y,z
    ious_all = [element for tupl in ious for element in tupl]
    devs_x_all = []
    devs_y_all = []
    devs_z_all = []
    for tuple in center_devs:
        for elem in tuple:
            dev_x, dev_y, dev_z = elem
            devs_x_all.append(dev_x)
            devs_y_all.append(dev_y)
            devs_z_all.append(dev_z)
    

    # compute statistics
    stdev__ious = np.std(ious_all)
    mean__ious = np.mean(ious_all)

    stdev__devx = np.std(devs_x_all)
    mean__devx = np.mean(devs_x_all)

    stdev__devy = np.std(devs_y_all)
    mean__devy = np.mean(devs_y_all)

    stdev__devz = np.std(devs_z_all)
    mean__devz = np.mean(devs_z_all)
    #std_dev_x = np.std(devs_x)

    # plot results
    data = [precision, recall, ious_all, devs_x_all, devs_y_all, devs_z_all]
    titles = ['detection precision', 'detection recall', 'intersection over union', 'position errors in X', 'position errors in Y', 'position error in Z']
    textboxes = ['', '', '',
                 '\n'.join((r'$\mathrm{mean}=%.4f$' % (np.mean(devs_x_all), ), r'$\mathrm{sigma}=%.4f$' % (np.std(devs_x_all), ), r'$\mathrm{n}=%.0f$' % (len(devs_x_all), ))),
                 '\n'.join((r'$\mathrm{mean}=%.4f$' % (np.mean(devs_y_all), ), r'$\mathrm{sigma}=%.4f$' % (np.std(devs_y_all), ), r'$\mathrm{n}=%.0f$' % (len(devs_x_all), ))),
                 '\n'.join((r'$\mathrm{mean}=%.4f$' % (np.mean(devs_z_all), ), r'$\mathrm{sigma}=%.4f$' % (np.std(devs_z_all), ), r'$\mathrm{n}=%.0f$' % (len(devs_x_all), )))]

    f, a = plt.subplots(2, 3)
    a = a.ravel()
    num_bins = 20
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    for idx, ax in enumerate(a):
        ax.hist(data[idx], num_bins)
        ax.set_title(titles[idx])
        if textboxes[idx]:
            ax.text(0.05, 0.95, textboxes[idx], transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', bbox=props)
    plt.tight_layout()
    plt.show()

