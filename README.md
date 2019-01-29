# OpenCV Leaf Angle Distribution Detection
The goal is to measure Rice Plant Leaf Angle Distribution using images. Given an image of a Rice Plant this will located Rice Plant leaves and determined its leaf angle distribution. Leaf Angle Distribution is imporatant as more erect leaves allow the leaves below to have more sunlight which affects the photosynthesis process. The end goal program is either a Web Applicaiton where images of plants could be uploaded that will return the Leaf Angle Distribution. 

Additional features and data regarding the rice plant may be added in the future. 

Terminal Output: 
![alt text](https://raw.githubusercontent.com/stephensamonte/OpenCV-Leaf-Angle-Distribution-Detection/master/Terminal%20Output/Terminal%20Output.PNG)

Example Result: ![alt_text](https://github.com/stephensamonte/OpenCV-Leaf-Angle-Distribution-Detection/blob/master/Archive/2019.01.20%20Curved%20HoughLines.jpg?raw=true)

# Notes: 
- Currently waiting new image data that is leveled with the ground. 

# My Experience / Context
This is my 3rd use of OpenCV for a project. My first attempt with OpenCV was a self Driving Remote Contorl Car and my second project with OpenCV was the HumanDetection Python Program. 

# Programming Environment: 
- This is a python program that is being modified in the [Python IDLE terminal](https://www.python.org/downloads/). This project utilizes [Python OpenCV](https://pypi.org/project/opencv-python/), and [numpy](http://www.numpy.org/). To run this project you will have to install python, python OpenCV, and numpy. 

## Installing Python 
- To install python follow the: [Python Windows Install](https://www.python.org/downloads/) 

## Installing Python OpenCV 
- To install OpenCV for python follow this: https://pypi.org/project/opencv-python/
- Verify that you have installed opencv-python by opening a python script shell (IDLE terminal) and runnign the following. IF correctly installed you should see what version of openCV you have installed. 
	Script: 
		>>> import cv2
		>>> cv2.__version__
	Result:
		>>> '3.4.5'

## Installing Python Numpy 
- To install numpy run `pip install numpy` in a terminal. 
- Verify that you have numpy installed by opening a python script shell (IDLE terminal) and running the following. If correctly installed you should see what verion of numpy you have installed.: 
	Script:
		>>> import numpy as np
		>>> np.__verion__
	Result: 
		>>> '1.13.1' 

# Run the program
- Open the .py file in the Python IDLE terminal and hit run. 

# References: 
- Leaf Angle Distribution Info & Importance: https://en.wikipedia.org/wiki/Leaf_angle_distribution#Examples_of_Leaf_Angle_Distributions


# Photo Reference:
- 1: http://jaiswallab.cgrb.oregonstate.edu/node?page=30 
- 2: https://newscenter.lbl.gov/2012/11/12/a-better-route-to-xylan/ 
- 3: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5313149/ 

- 10: Erect_Leaves - Provided by Omar Samonte
- 11: Droopy_Leaves - Provided by Omar Samonte


# Journal 
- 2018.12.30 Programming project was conceptualized by Omar Samonte
- 2018.12.31 Research on how plant leave angles are measured. I found out that Leaf Angle Distribution is what I am trying to measure and that it is a time consuming process so data on Leaf Angle Distribution is scarce. The current way Leaf Angle Distribution is given to plants is by sight and researcher sound judgement to assign a classification to the plant. This is not quantifiable. I could not find any program that automatically measures leaf angle distribution. Physically there is a tool that researchers use to trace all the leaves of a plant to determing the amount of curve of the leaf. Through photo a researcher has to manually calculate the angles of the leaves to determing the distribution of the angle. 
- 2019.01.01 Research on OpenCV and it's capabilities. 
- 2019.01.01 Improvements to work with test rice plant images online 
- 2019.01.02 Improvemnents to work with real usecase rice plant images. I was given images by Omar Samonte of Rice Plants that he wishes to use the program for. Problems: There's so many things going on in the backgorund. So, I looked into ignoring the background. I added function to filter the image based on the color of the plant. I could train a Tensor Flow Model to mask the plant but it is currently unnecessary. 
- 20199.01.29 Added how to set up Programming environment in README.md

# Project: 
- 2018.12.31 Email from Omar: I was wondering if each picture could be converted into line segments, and then the average angle of all line segments can be computed. This average angle would be useful in  evaluating the rice that we grow in our field experiments. In rice breeding where we evaluate thousands of different rice rows, we prefer a plant type that has erect leaves and stems because this plant type is more efficient in photosynthesis and it also is less prone to lodging (or plants leaning and falling down)
