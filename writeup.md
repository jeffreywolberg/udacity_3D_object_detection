# Writeup: Track 3D-Objects Over Time

You can find the root mean squared error graphs for each part in the folder `rmse_graphs`

Moreover, you can find the tracking results movie in the file   `my_tracking_results.mp4`

Please use this starter template to answer the following questions:

### 1. Write a short recap of the four tracking steps and what you implemented there (filter, track management, association, camera fusion). Which results did you achieve? Which part of the project was most difficult for you to complete, and why?

Filter - I implemented an Extended Kalman Filter to fuse prediction and measurements. 

Track Management - I wrote the logic to determine whether a track was 'initialized', 'tentative', or 'confirmed' based on the track score. Moreover, I implemented the logic to determine whether to delete a track.

Association - I created an association matrix with the computed Mahalanobis distances. I then determined the min distance and associated a measurement with a track.

Camera Fusion - I incorporated camera measurements into the sensor fusion model, providing the kalman filter with another useful measurements when doing filtering


### 2. Do you see any benefits in camera-lidar fusion compared to lidar-only tracking (in theory and in your concrete results)? 

Redundancy of sensors is very important when trying to understand the world around you.

In my case, a LiDAR measurement would occasionally return a false positive measurement but it would be negated by the camera measurement that did not include that false positive.


### 3. Which challenges will a sensor fusion system face in real-life scenarios? Did you see any of these challenges in the project?

A sensor-fusion system will face challenges such as properly weeding out false positives/negatives, and having the sensors be robust to all different types of scenarios. 

### 4. Can you think of ways to improve your tracking results in the future?

Yes. We can implement more sophisticated data association, using global nearest neighbor or join probabilistic data association. Moreover, instead of assuming linear motion of cars in any direction, we can make better assumptions about the motion of our cars (they will mostly go forwards or backwards relative to the vantage point).

