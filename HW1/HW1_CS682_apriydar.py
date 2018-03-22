###############################
# Aakarshika Priydarshi
# G01047156
# CS 682
# Homework 1
###############################


import numpy as np
import cv2

font = cv2.FONT_HERSHEY_SIMPLEX
img= cv2.imread("originals/org.jpg",-1)

# gray= cv2.imread("original.png",0)

###########################################
# 1. Color Plane Switch
###########################################
b,g,r = cv2.split(img)
img1 = cv2.merge((g,b,r))

# cv2.imshow("T1- Switch Color Plane",img1)


###########################################
# 2. Color transformations
###########################################

img2a = cv2.imread("originals/org.jpg",-1)
img2a[:,:,0] = 100
# cv2.imshow("T2.a.- Blue value of 100 ",img2a)

img2b = cv2.imread("originals/org.jpg",-1)
img2b[:,:,0] = 0
# cv2.imshow("T2.b.- Blue value of 0 ",img2b)



cv2.putText(img,'Original', (30,65), font, 1.5, (150,5,5), 3, cv2.LINE_AA)
cv2.putText(img1,'RGB -> RBG', (30,65), font, 1.5, (150,5,5), 3, cv2.LINE_AA)
cv2.putText(img2a,'Blue value of 100.', (30,65), font, 1.5, (150,5,5), 3, cv2.LINE_AA)
cv2.putText(img2b,'Blue value of 0', (30,65), font, 1.5, (150,5,5), 3, cv2.LINE_AA)

combo3= np.hstack((img,img1,img2a,img2b))
cv2.namedWindow('Color Transformations',cv2.WINDOW_NORMAL)
cv2.resizeWindow('Color Transformations', 1400,700)
cv2.imshow('Color Transformations',combo3)


###########################################
# 3. Adding a hat to the face
###########################################


cascPath = 'haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascPath)

hat = cv2.imread('originals/hat2.png') #hat image
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

img3=cv2.imread("originals/org.jpg",-1) #copy

faces = faceCascade.detectMultiScale(
	gray,
	scaleFactor=1.1,
	minNeighbors=5,
	minSize=(160, 160),
	flags=cv2.CASCADE_SCALE_IMAGE
)


rowsImg3,colsImg3,channels1 = img3.shape

# Detect faces and get face co-ordinates
for (x, y, w, h) in faces:
	h=h-h/3
	w=w+w/6
	starty=y-h
	endy=y+h/3
	startx=x-w/4
	endx=x+w+w/4
	if startx<0:
	    startx=0
	if starty<0:
	    starty=0
	if endx>colsImg3:
	    endx=colsImg3
	if endy>rowsImg3:
	    endy=rowsImg3

	# adjust hat location and size
	hatImg = cv2.resize(hat, (endx-startx, endy-starty))
	hatImggray = cv2.cvtColor(hatImg,cv2.COLOR_BGR2GRAY)
	ret, mask = cv2.threshold(hatImggray, 10, 255, cv2.THRESH_BINARY)
	mask_inv = cv2.bitwise_not(mask)
	rows,cols,channels = hatImg.shape

	# remove hat part from background image
	bg = img3[starty:endy,startx:endx]
	Img3_bg = cv2.bitwise_and(bg,bg,mask = mask_inv)

	# mask black area from hat
	hatImg_fg = cv2.bitwise_and(hatImg,hatImg,mask = mask)

	# Put hat on image
	final = cv2.add(Img3_bg,hatImg_fg)
	img3[starty:endy,startx:endx] = final

cv2.putText(img3,'Using masks and face detection.', (30,65), font, 1.5, (150,5,5), 3, cv2.LINE_AA)

cv2.namedWindow('Hat',cv2.WINDOW_NORMAL)
cv2.resizeWindow('Hat', 1400,700)
cv2.imshow('Hat', img3)


###########################################
# 4. Filters:
###########################################

img41=cv2.imread("originals/org_with_noise.png",-1)
kernel = np.ones((5,5),np.float32)/25
img4a = cv2.filter2D(img41,-1,kernel)
# cv2.imshow('T4- Filter2D Convolution 5x5', img4a)

img42=cv2.imread("originals/org_with_noise.png",-1)
img4b = cv2.medianBlur(img42,5)
# cv2.imshow('T4- Median Blur 5x5', img4b)


cv2.putText(img41,'Original (30 percent Noise Added)',(30,65), font, 1.5, (150,5,5), 3, cv2.LINE_AA)
cv2.putText(img4a,'Filter2D Convolution 5x5', (30,65), font, 1.5, (150,5,5), 3, cv2.LINE_AA)
cv2.putText(img4b,'Median Blur 5x5.',(30,65), font, 1.5, (150,5,5), 3, cv2.LINE_AA)

combo3= np.hstack((img41,img4a,img4b))
cv2.namedWindow('Blurr Filters',cv2.WINDOW_NORMAL)
cv2.resizeWindow('Blurr Filters', 1400,700)
cv2.imshow('Blurr Filters',combo3)





###########################################
# 5. Extract specific colored part of image
###########################################


#  range of color limits in HSV
lower_blue = np.array([110,50,50])
upper_blue = np.array([140,255,255])

lower_green = np.array([30,40,40])
upper_green = np.array([70,220,220])


img5 = cv2.imread("originals/org_colors.jpg",-1)
# Convert BGR to HSV
hsv = cv2.cvtColor(img5, cv2.COLOR_BGR2HSV)

# Thresholding hsv image
mask_b = cv2.inRange(hsv, lower_blue, upper_blue)
mask_b_inv = cv2.bitwise_not(mask_b)
# Bitwise-AND mask and original image
onlyBlue = cv2.bitwise_and(img5,img5, mask= mask_b )

# Thresholding hsv image
mask_g = cv2.inRange(hsv, lower_green, upper_green)
mask_g_inv = cv2.bitwise_not(mask_g)
repG2R = cv2.bitwise_and(img5,img5, mask= mask_g_inv )

px=np.where((repG2R == [0,0,0] ).all(axis = 2));
repG2R[px] = [50,50,200]

cv2.putText(img5,'Original', (30,65), font, 1.5, (150,5,5), 3, cv2.LINE_AA)
cv2.putText(onlyBlue,'Blue color extracted', (30,65), font, 1.5, (150,5,5), 3, cv2.LINE_AA)
cv2.putText(repG2R,'Green color in range replaced by constant red.', (30,65), font, 1, (150,5,5), 3, cv2.LINE_AA)

combo2= np.hstack((img5,onlyBlue,repG2R))

cv2.namedWindow('Extract specific colored part of image',cv2.WINDOW_NORMAL)
cv2.resizeWindow('Extract specific colored part of image', 1400,700)
cv2.imshow('Extract specific colored part of image',combo2)


# cv2.imshow('T5- Only blue colored parts',onlyBlue)
# cv2.imshow('T5- Replace range of green with constant red',repG2R)




###########################################
# 6. Edge detection in grayscale
###########################################

# compute the median of the single channel pixel intensities
gray = cv2.imread("originals/org.jpg",0)

sigma=0.33
v = np.median(gray)
lower = int(max(0, (1.0 - sigma) * v))
upper = int(min(255, (1.0 + sigma) * v))
canny = cv2.Canny(gray,lower,upper)

blurred= cv2.GaussianBlur(gray,(3,3),0)
laplacian = cv2.Laplacian(blurred,cv2.CV_8U)

clahe = cv2.createCLAHE(clipLimit=8, tileGridSize=(8,8))
cl1 = clahe.apply(laplacian)

cv2.putText(canny,'Canny',(30,65), font, 1.5, (150,5,5), 3, cv2.LINE_AA)
cv2.putText(cl1,'Laplacian filter and CLAHE contrast adjustment', (30,65), font, 1, (150,5,5), 3, cv2.LINE_AA)
cv2.putText(gray,'Original',(30,135), font, 1.5, (150,5,5), 3, cv2.LINE_AA)
combo= np.hstack((gray,canny,cl1))

cv2.putText(combo,'Press "q" to quit, "s" to save ', (365,30), font, 1, (150,5,5), 3, cv2.LINE_AA)

cv2.namedWindow('Edge detection',cv2.WINDOW_NORMAL)
cv2.resizeWindow('Edge detection', 1400,700)
cv2.imshow('Edge detection',combo)

k = cv2.waitKey(0)
if k == ord('q'): # wait for 'q' key to exit 
	cv2.destroyAllWindows()
elif k == ord('s'): # wait for 's' key to save and exit 
	cv2.imwrite('results/Color_Transformation_1.png',img1)
	cv2.imwrite('results/Color_Transformation_2.png',img2a)
	cv2.imwrite('results/Color_Transformation_3.png',img2b)
	cv2.imwrite('results/Filter2D_1.png',img4a)
	cv2.imwrite('results/MedianBlurFilter_2.png',img4b)
	cv2.imwrite('results/Hat.png',img3)
	cv2.imwrite('results/Color_Extract.png',onlyBlue)
	cv2.imwrite('results/Color_Exchange.png',repG2R)
	cv2.imwrite('results/Grayscale.png',gray)
	cv2.imwrite('results/Edge_detection_Canny.png',canny)
	cv2.imwrite('results/Edge_detection_Laplacian.png',cl1)

	cv2.destroyAllWindows()

