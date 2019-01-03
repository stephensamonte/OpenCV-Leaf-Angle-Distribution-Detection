# OpenCV Leaf Angle Distribution Detection
The goal is to measure Rice Plant Leaf Angle Distribution using images. Leaf Angle Distribution is imporatant as more erect leaves allow the leaves below to have more sunlight which affects the photosynthesis process. The end goal program is either a Web Applicaiton where images of plants could be uploaded that will return the Leaf Angle Distribution. 

Additional features and data regarding the rice plant may be added in the future. 


Leaf Angle Distribution Info & Importance: 
https://en.wikipedia.org/wiki/Leaf_angle_distribution#Examples_of_Leaf_Angle_Distributions

# My Experience / Context
This is my 3rd use of OpenCV for a project. My first attempt with OpenCV was a self Driving Remote Contorl Car and my second project with OpenCV was the HumanDetection Python Program. 


# Photo Reference:
1: http://jaiswallab.cgrb.oregonstate.edu/node?page=30 
2: https://newscenter.lbl.gov/2012/11/12/a-better-route-to-xylan/ 
3: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5313149/ 

10: Erect_Leaves - Provided by Omar Samonte
11: Droopy_Leaves - Provided by Omar Samonte


# Journal 
2018.12.30 Programming project was conceptualized by Omar Samonte

2018.12.31 Research on how plant leave angles are measured. I found out that Leaf Angle Distribution is what I am trying to measure and that it is a time consuming process so data on Leaf Angle Distribution is scarce. The current way Leaf Angle Distribution is given to plants is by sight and researcher sound judgement to assign a classification to the plant. This is not quantifiable. I could not find any program that automatically measures leaf angle distribution. Physically there is a tool that researchers use to trace all the leaves of a plant to determing the amount of curve of the leaf. Through photo a researcher has to manually calculate the angles of the leaves to determing the distribution of the angle. 

2019.01.01 Research on OpenCV and it's capabilities. 

2019.01.01 Improvements to work with test rice plant images online 

2019.01.02 Improvemnents to work with real usecase rice plant images. I was given images by Omar Samonte of Rice Plants that he wishes to use the program for. Problems: There's so many things going on in the backgorund. So, I looked into ignoring the background. I added function to filter the image based on the color of the plant. I could train a Tensor Flow Model to mask the plant but it is currently unnecessary. 

# Project: 
2018.12.31 Email from Omar: 
I was wondering if each picture could be converted into line segments, and then the average angle of all line segments can be computed. This average angle would be useful in  evaluating the rice that we grow in our field experiments. In rice breeding where we evaluate thousands of different rice rows, we prefer a plant type that has erect leaves and stems because this plant type is more efficient in photosynthesis and it also is less prone to lodging (or plants leaning and falling down)
