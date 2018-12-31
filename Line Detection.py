import math
import numpy as np
import cv2

# Read image
img = cv2.imread('./Photos/3.jpg')

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
#lines = cv2.HoughLines(edges,4,np.pi/180, 80)
lines = cv2.HoughLinesP(edges, 1, np.pi/180, 85, maxLineGap=40)

for line in lines:
    x1, y1, x2, y2 = line[0]
    # add lines to image
    cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)


#lines = cv2.HoughLines(edges,1,np.pi/180,200)
#lines = cv2.HoughLines(edges, 1, np.pi / 180, 100, None, 0, 0)

#for i in range(0, len(lines)):
#    rho = lines[i][0][0]
#    theta = lines[i][0][1]
#    a = math.cos(theta)
#    b = math.sin(theta)
#    x0 = a * rho
#    y0 = b * rho
#    pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
#    pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
#    cv2.line(img, pt1, pt2, (255,0,0), 3, cv2.LINE_AA)

cv2.imshow("Image lines", img)
      
# All the changes made in the input image are finally 
# written on a new image houghlines.jpg 
# cv2.imwrite('linesDetected.jpg', img) 

# Line Detection
#rho = 50 # distance resolution in pixels of the Hough grid
#theta = np.pi/180 # angular resolution in radians of the Hough grid
#threshold = 80     # minimum number of votes (intersections in Hough grid cell)
#min_line_length = 100 #minimum number of pixels making up a line
#max_line_gap = 10    # maximum gap in pixels between connectable line segments
#line_image = np.copy(img)*0 # creating a blank to draw lines on


# Get Detected lines 
#lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),min_line_length, max_line_gap)

# lines_edges = cv2.addWeighted(img, 0.8, line_image, 1, 0)
# cv2.imshow('lines_edges image',lines_edges)

k = cv2.waitKey(0)
if k == 27:         # wait for ESC key to exit
    cv2.destroyAllWindows()
elif k == ord('s'): # wait for 's' key to save and exit
    cv2.imwrite('test.png',img)
    cv2.destroyAllWindows()
