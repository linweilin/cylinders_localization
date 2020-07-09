# cylinders_localization
## A registration algorithm with line features

## 1. Prerequisites 
1.1 Make sure **Ceres version >= 1.12**.

## 2. Build cylinders_localization on ROS
1.1 Clone the repository and catkin_make:
```
    cd ~/catkin_ws/src
    git clone https://github.com/linweilin17/cylinders_localization.git
```

1.2 Modify **parameter_reader.h** in the directory **include**

```
string filename="YOUR_PATH_TO_PACKAGE/cylinders_localization/parameters.txt"
```

1.3 Catkin make

```
    cd ../
    catkin_make
    source ~/catkin_ws/devel/setup.bash
```

## 3. Download the datasets in pan.baidu.com

```
https://pan.baidu.com/s/10s3b-TK08dTx7vCz7Z6j7A 
提取码：3sbm
```

## 4. To execute the program:
4.1 Open three terminal:

- run `$ rosrun cylinders_localization cylinders_localization` **in YOUR_PATH_TO_PACKAGE**.
- run `$ roslaunch cylinders_localization cylinders_localization.launch` to open rviz.
- run `$ rosbag play -l YOUR_PATH_TO_DATASET/truss_328_3_poles.bag`.
    
## NOTE

- The start point and end point of lines in bim_vector_map.txt must be in **acsending** order.
