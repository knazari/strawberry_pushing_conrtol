import cv2
import numpy as np
import matplotlib
from matplotlib.pyplot import imshow
from matplotlib import pyplot as plt

filein = "/home/kiyanoush/Cpp_ws/src/haptic_finger_control/RT-Data/reactive/22.png"
img = cv2.imread(filein).astype(np.uint8)
img[250:, :, :] = 0
img[230:254, 150:230, :] = 0


# Thresholding the gray image to find sensor markers
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
median = cv2.medianBlur(gray, 5)
blurred = cv2.GaussianBlur(median, (5, 5), 0)
(T, threshold) = cv2.threshold(blurred, 72, 255, cv2.THRESH_BINARY)

# while True:
#     cv2.imshow("img", img)
#     cv2.imshow("gray", gray)
#     cv2.imshow("median", median)
#     cv2.imshow("blurred", blurred)
#     cv2.imshow("threshold", threshold)
#     if cv2.waitKey(1) & 0xFF == 27:
#         break
# cv2.destroyAllWindows()

th3 = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 13, 2.4)

# Detecting the countours and find pixel coordinates of the sensor markers
contours, hierarchy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# threshold = cv2.cvtColor(threshold, cv2.COLOR_GRAY2BGR)
# th3 = cv2.cvtColor(th3, cv2.COLOR_GRAY2BGR).astype(np.uint8)

# cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
# #  select the first contour
# for cnt in contours:
#     # find the extreme points
#     leftmost   = tuple(cnt[cnt[:,:,0].argmin()][0])
#     rightmost  = tuple(cnt[cnt[:,:,0].argmax()][0])
#     topmost    = tuple(cnt[cnt[:,:,1].argmin()][0])
#     bottommost = tuple(cnt[cnt[:,:,1].argmax()][0])
#     points = [leftmost, rightmost, topmost, bottommost]

#     # draw the points on th image
#     for point in points:
#         cv2.circle(threshold, point, 4, (0, 0, 255), -1)

# # display the image with drawn extreme points
# while True:
#     cv2.imshow("Extreme Points", threshold)
#     if cv2.waitKey(1) & 0xFF == 27:
#         break
# cv2.destroyAllWindows()

# # cv2.imwrite("contoursimple.png", th3)
# # cv2.imwrite("threshold_adaptive.png", th3)
import imutils
cnts = cv2.findContours(threshold.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
threshold = cv2.cvtColor(threshold, cv2.COLOR_GRAY2BGR)
for c in cnts:
	# compute the center of the contour
	M = cv2.moments(c)
	if M["m00"] != 0:
		cX = int(M["m10"] / M["m00"])
		cY = int(M["m01"] / M["m00"])
		# draw the contour and center of the shape on the image
		# cv2.drawContours(img, [c], -1, (0, 255, 0), 2)
		cv2.circle(threshold, (cX, cY), 10, (0, 255, 0), 2)
		# cv2.putText(img, "center", (cX - 20, cY - 20),
		# 	cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

cv2.imwrite("bestcontourthresholddeformed2.png", threshold[:300])
while True:
	cv2.imshow("gray1", threshold)
	# cv2.imshow("gray2", th3)
	if cv2.waitKey(1) & 0xFF == 27:
		break
cv2.destroyAllWindows()
