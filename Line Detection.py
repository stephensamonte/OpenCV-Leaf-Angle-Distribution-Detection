import math
import numpy as np
import cv2

#def centerOfLines():


def maskPlantByColor(image):
    # Make a copy of the image
    im = np.copy(originalImage)

    ## convert to hsv
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    ## mask of green (36,0,0) ~ (70, 255,255)
    mask1 = cv2.inRange(hsv, (36, 0, 0), (70, 255,255))

    ## mask o yellow (15,0,0) ~ (36, 255, 255)
    #mask2 = cv2.inRange(hsv, (15,0,0), (36, 255, 255))

    ## final mask and masked
    #mask = cv2.bitwise_or(mask1, mask2)
    target = cv2.bitwise_and(im,im, mask=mask1)

    cv2.imwrite("target.png", target)
    return target

    
def getEdges(blur_gray):    
    # Canny Edge Detection
    low_threshold = 150
    high_threshold = 250
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold, apertureSize = 3)

    # Disaply Blurred Image
    #cv2.imshow('Edges image',edges)
    #Save Image
    cv2.imwrite("Edges.jpg", edges)
    return edges
    
# URL: https://docs.opencv.org/3.4/db/d5c/tutorial_py_bg_subtraction.html
def getCountour(image):
    # Retrieve photo 

    # Make a copy of the image
    im = np.copy(originalImage)
    #imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(image, 127, 255, 0)
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #extract contour data
    cnt = contours[0]
    M = cv2.moments(cnt)
    print("Moment", M)    
    
    cv2.drawContours(im, contours, -1, (255,0,0), 3)
    cv2.imwrite("contour.jpg", im)
    #cv2.imshow('contour',hierarchy)
    #return contours


def extractLines(edges):
    # Line Detection
    # This returns an array of r and theta values 
    #lines = cv2.HoughLines(edges,4,np.pi/180, 80)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 85, maxLineGap=40)

    angle = 0.0;
    count_stem_lines = 0

    for line in lines:
        x1, y1, x2, y2 = line[0]
        # add lines to image
        #cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
    

    #cv2.imshow("Image lines", img)
    cv2.imwrite("verticalLines.jpg", vertical)
    
def verticalLines(image):

    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 15)
    
    # Copy image to extract vertical lines from
    vertical = np.copy(thresh)
    
    # Specify size on vertical axis
    rows = vertical.shape[0]
    verticalsize = math.ceil(rows / 10)

    # Create structure element for extracting vertical lines through morphology operations
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize))
    
 
    # Apply morphology operations
    vertical = cv2.erode(vertical, verticalStructure)
    vertical = cv2.dilate(vertical, verticalStructure)
    print(vertical)

    # Show extracted vertical lines
    cv2.imwrite("vertical.jpg", vertical)
    

    # sum columns. Find the 2 columns with the most 1s
    #for i in range(len(vertical)):
    #    for j in range(len(vertical[i])):
    #        print(vertical[i][j])

    #cv2.imshow('mmmmm1',thresh)
    thresh-=vertical
    #cv2.imshow('mmmmm',thresh)

    cv2.imwrite("noVerticalThresh.jpg", thresh)
    
    
def horizontalLines(image):
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 15)
    
    # Create the images that will use to extract the horizontal and vertical lines
    horizontal = np.copy(thresh)

    # Specify size on horizontal axis
    cols = horizontal.shape[1]
    horizontal_size = math.ceil(cols / 20)

    # Create structure element for extracting horizontal lines through morphology operations
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))

    # Apply morphology operations
    horizontal = cv2.erode(horizontal, horizontalStructure)
    horizontal = cv2.dilate(horizontal, horizontalStructure)

    # Show extracted horizontal lines
    cv2.imwrite("horizontal.jpg", horizontal)
    
    
    
def radToDegree(val):
    #print(np.rad2deg(val))
    return np.rad2deg(val)
    
# Read image
originalImage = cv2.imread('./Photos/11.jpg')

#filter image by color 
#filterImage = maskPlantByColor(originalImage)
filterImage = originalImage

# Locate the Horizontal and Vertical pixels
#horizontalLines(originalImage)
#verticalLines(originalImage)

# Convert Image to Grayscale
grey_image = cv2.cvtColor(filterImage, cv2.COLOR_BGR2GRAY)

# Blur Image to remove noise
kernel_size = 5
blur_gray = cv2.GaussianBlur(grey_image,(kernel_size, kernel_size),0)
#Save Image
cv2.imwrite("blur_gray.jpg", blur_gray)

# Find Edges
edges = getEdges(blur_gray)

#cv2.imshow('Edges image',edges)

# Find contours
#getCountour(blur_gray)


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
minLineLength = 10
maxLineGap=40
threshold = 50
## cv2.HoughLinesP(image, rho, theta, threshold[, lines[, minLineLength[, maxLineGap]]])
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, None, 10, 1)

angle = 0.0
count_stem_lines = 0
sumAngle = 0.0

# Perform action on each line
for line in lines:    
    x1, y1, x2, y2 = line[0]
    # add lines to image
    cv2.line(originalImage, (x1, y1), (x2, y2), (255, 0, 0), 10)

    # calculate radian angle of line to horizontal
    radianA = math.atan2(abs(y1 - y2), abs(x1 - x2))

    # Calculate angle in degree to horizontal
    angleHor = np.rad2deg(radianA)
    angle = angleHor 
    
    #print("Hello Rice", angle)
    sumAngle += angle;

    # Determine the number of line segments
    count_stem_lines += 1 
    

    #print(angle)
    
    """
    #print(deg)
    if (deg > 91.0 or deg < 89.0):
        convert = deg - 90
        print(deg-90)
    else:
        count_stem_lines = count_stem_lines + 1 
        print("ROOT StemLines: ", count_stem_lines)
    """

# Calculate the average angle 
averageAng = sumAngle / count_stem_lines
print("sum: ", sumAngle)
print("StemLines: ", count_stem_lines)
print("averageAng: ", averageAng)

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

#cv2.imshow("Image lines", img)
#Save Image
cv2.imwrite("HoughLines.jpg", originalImage)
      
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


# Original Image
#cv2.imshow('image',img)
# Gray Scale Image
# cv2.imshow('Gray image',grey_image)
# Disaply Blurred Image
#cv2.imshow('Gray image',blur_gray)
