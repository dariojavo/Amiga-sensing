import numpy as np
import cv2
import struct

image_path = '/mnt/data_recording/Amiga_record_1692402384/oak1/rgb_image_1692403011.2719598.txt'

with open(image_path, 'rb') as f:
    # Read the metadata
    width, height, channels, dtype_len = struct.unpack('iiii', f.read(16))
    dtype_str = f.read(dtype_len).decode()
    dtype = np.dtype(dtype_str)

    # Read the image data
    img_array = np.frombuffer(f.read(), dtype=dtype).reshape((height, width, channels))

    desired_width = 720
    desired_height = 480

    # Resize the image
    resized_img = cv2.resize(img_array, (desired_width, desired_height))

    # Display the resized image
    cv2.imshow('Resized Image', resized_img)
    cv2.waitKey(5000)  # Pause for 5 seconds
    cv2.destroyAllWindows()