import cv2
import os
from cv2 import imread
import numpy as np


paths = []
for root, dir, files in os.walk('/usr/xtmp/ct214/daml/vr_sickness/flownet2-pytorch/skyhouse_perspective_png_output/'):
    for file in files:
        if file.endswith(".png"):
            path = os.path.join(root, file)
            paths.append(path)
paths.sort()

video_name = "/usr/xtmp/ct214/daml/vr_sickness/test1025.mp4"
video = cv2.VideoWriter(str(video_name), cv2.VideoWriter_fourcc(*'mp4v'), 30, (imread(paths[0]).shape[1], imread(paths[0]).shape[0]))
images = [imread(path) for path in paths]
count = 0
print(len(images))
for image in images:
    video.write(image)
    count += 1
    print(count)

video.release()