#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author : Gabriele Gandolfi


import io
import socket
import struct
from PIL import Image
Image2 = Image
import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# Start a socket listening for connections on 0.0.0.0:8000 (0.0.0.0 means
# all interfaces)
server_socket = socket.socket()
server_socket.bind(('169.254.214.66', 8000))
server_socket.listen(0)

# Accept a single connection and make a file-like object out of it
connection = server_socket.accept()[0].makefile('rb')

# result = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'MJPG'), 60, (640, 480))

rospy.init_node('tactile_finger', anonymous=False)
img_pub = rospy.Publisher('/fing_camera/color/image_raw', Image, queue_size=1)
bridge = CvBridge()

try:
    while True:
        # Read the length of the image as a 32-bit unsigned int. If the
        # length is zero, quit the loop
        image_len = struct.unpack('<L', connection.read(struct.calcsize('<L')))[0]
        if not image_len:
            break
        # Construct a stream to hold the image data and read the image
        # data from the connection
        image_stream = io.BytesIO()
        image_stream.write(connection.read(image_len))
        # Rewind the stream, open it as an image with PIL and do some
        # processing on it
        image_stream.seek(0)
        image = Image2.open(image_stream)
        cv_image = np.array(image)
        cv2.imshow('Stream',cv_image)

        ros_color_image = bridge.cv2_to_imgmsg(np.array(cv_image), "rgb8")
        ros_color_image.header.stamp = rospy.Time.now()
        img_pub.publish(ros_color_image)

        # result.write(cv_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    connection.close()
    server_socket.close()
