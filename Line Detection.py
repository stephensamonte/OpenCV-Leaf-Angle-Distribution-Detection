import numpy as np
import cv2

# Read image
img = cv2.imread('2.jpg')

# Convert Image to Grayscale
grey_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Original Image
#cv2.imshow('image',img)

# Gray Scale Image
# cv2.imshow('Gray image',grey_image)


# Blur Image to remove noise
kernel_size = 5
blur_gray = cv2.GaussianBlur(grey_image,(kernel_size, kernel_size),0)

# Disaply Blurred Image
cv2.imshow('Gray image',blur_gray)


# Canny Edge Detection
low_threshold = 50
high_threshold = 150
edges = cv2.Canny(blur_gray, low_threshold, high_threshold, apertureSize = 3)

# Disaply Blurred Image
cv2.imshow('Edges image',edges)


# Shape of region where to find lines
#imshape = img.shape # Determine image size
# blank mask:
#mask = np.zeros_like(img)

# Trapezoid
#vertices = np.array([[(0,imshape[0]),(450, 320), (500, 320), (imshape[1],imshape[0])]], dtype=np.int32)
#cv2.fillPoly(mask, vertices, 255)
#masked_edges = cv2.bitwise_and(edges, mask)

# Line Detection
# This returns an array of r and theta values 
lines = cv2.HoughLines(edges,1,np.pi/180, 80) 
  
# The below for loop runs till r and theta values  
# are in the range of the 2d array 
for r,theta in lines[0]: 
      
    # Stores the value of cos(theta) in a 
    a = np.cos(theta) 
  
    # Stores the value of sin(theta) in b 
    b = np.sin(theta) 
      
    # x0 stores the value rcos(theta) 
    x0 = a*r 
      
    # y0 stores the value rsin(theta) 
    y0 = b*r 
      
    # x1 stores the rounded off value of (rcos(theta)-1000sin(theta)) 
    x1 = int(x0 + 1000*(-b)) 
      
    # y1 stores the rounded off value of (rsin(theta)+1000cos(theta)) 
    y1 = int(y0 + 1000*(a)) 
  
    # x2 stores the rounded off value of (rcos(theta)+1000sin(theta)) 
    x2 = int(x0 - 1000*(-b)) 
      
    # y2 stores the rounded off value of (rsin(theta)-1000cos(theta)) 
    y2 = int(y0 - 1000*(a)) 
      
    # cv2.line draws a line in img from the point(x1,y1) to (x2,y2). 
    # (0,0,255) denotes the colour of the line to be  
    #drawn. In this case, it is red.  
    cv2.line(img,(x1,y1), (x2,y2), (0,0,255),2) 
      
# All the changes made in the input image are finally 
# written on a new image houghlines.jpg 
cv2.imwrite('linesDetected.jpg', img) 



# Line Detection
# rho = 50 # distance resolution in pixels of the Hough grid
# theta = np.pi/180 # angular resolution in radians of the Hough grid
# threshold = 80     # minimum number of votes (intersections in Hough grid cell)
# min_line_length = 100 #minimum number of pixels making up a line
# max_line_gap = 10    # maximum gap in pixels between connectable line segments
# line_image = np.copy(img)*0 # creating a blank to draw lines on


# # Get Detected lines 
# lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),min_line_length, max_line_gap)

# # Draw lines onto blank image 
# for line in lines:
#     for x1,y1,x2,y2 in line:
#         cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),2)

# lines_edges = cv2.addWeighted(img, 0.8, line_image, 1, 0)
# cv2.imshow('lines_edges image',lines_edges)

k = cv2.waitKey(0)
if k == 27:         # wait for ESC key to exit
    cv2.destroyAllWindows()
elif k == ord('s'): # wait for 's' key to save and exit
    cv2.imwrite('messigray.png',img)
    cv2.destroyAllWindows()
