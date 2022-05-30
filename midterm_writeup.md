# Midterm Writeup

## Pictures of various car point clouds
![.](car_images/car1.jpg)
![.](car_images/car2.jpg)
![.](car_images/car3.jpg)
![.](car_images/car4.jpg)
![.](car_images/car5.jpg)
![.](car_images/car6.jpg)
![.](car_images/car7.jpg)
![.](car_images/car8.jpg)
![.](car_images/car9.jpg)
![.](car_images/car10.jpg)

- The vehicle features that appear stable are the sides/corners of the car that are facing the LiDAR sensor. They mostly include tail-lights and side/rear bumpers. Windows are often not included in the point clouds, because of low reflectivity, while the metallic body of the car is almost always included.

Here is the normalized intensity map for one of the photos. It demonstrates that the car bumpers are the part of the car that the LiDAR captures very well.
![.](car_images/intensity_map.jpg)

Here are the data plots for the detection of vehicles using the LiDAR sensor data:
Precision = 96.41%
Recall = 79.08%
![.](recall_precision_stats.jpg)
 