import os
import sys
import cv2
import numpy as np
from matplotlib.pyplot import imread, imsave
import time

class Enquirec2Perspec:
    # _init_ function
    # Input: 1. img: current frame of the panoramic video
    def __init__(self, img):
        self._img = img
        [self._height, self._width, _] = self._img.shape
    
    # Function convert equirectangular image to perspective
    # view based on the view position.
    # Input: 
    # 1. wFOV: horizontal field of view in degrees
    # 2. THETA: left/right angle in degrees of view center(right direction is positive, left direction is negative)
    # 3. PHI: up/down angle in degrees of view center(up direction is positive, down direction is negative)
    # 4. height, width: height/width of the output viewport image, should fit the resolution of each eye's viewport
    def GetPerspective(self, FOV, THETA, PHI, height, width):
        # set radius for the sphere which 
        # the equirectangular image is wrapped to
        RADIUS = 1
        
        # height, width of the input frame
        equ_h = self._height
        equ_w = self._width

        # center of the input frame
        equ_cx = (equ_w - 1) / 2.0
        equ_cy = (equ_h - 1) / 2.0
        # set vertical field of view based 
        # on output image size and horizontal 
        # field of view
        wFOV = FOV
        hFOV = float(height) / width * wFOV
        
        # center of the output image
        c_x = (width - 1) / 2.0
        c_y = (height - 1) / 2.0

        # horizontal length of view:
        # w_len = 2 * radius * tan(wFOV/2)
        w_len = 2 * RADIUS * np.tan(np.radians(wFOV / 2.0))
        # each pixels from frame represents 
        # how many units of the horizontal length of view
        w_interval = w_len / (width - 1)

        # same precedure for viertical view
        h_len = 2 * RADIUS * np.tan(np.radians(hFOV / 2.0)) 
        h_interval = h_len / (height - 1)
        
        # x_map: radius, distance between the viewport to the sphere center
        # y_map: horizontal distance between each image pixel and image center
        # z_map: vertical distance between each image pixel and image center
        x_map = np.zeros([height, width], np.float32) + RADIUS
        y_map = np.tile((np.arange(0, width) - c_x) * w_interval, [height, 1])
        z_map = -np.tile((np.arange(0, height) - c_y) * h_interval, [width, 1]).T
        # distance between the sphere center and each pixel at viewport image\
        # D = sqrt(radius^2 + horizontal_distance^2 + vertical_distance^2)
        D = np.sqrt(x_map**2 + y_map**2 + z_map**2)
        xyz = np.zeros([height, width, 3], np.float)
        # normalize to the sphere that equirectangular image is wrapped to
        xyz[:, :, 0] = (RADIUS / D * x_map)[:, :]
        xyz[:, :, 1] = (RADIUS / D * y_map)[:, :]
        xyz[:, :, 2] = (RADIUS / D * z_map)[:, :]
        
        # unit vector along vertical rotation axis
        vertical_axis = np.array([0.0, 1.0, 0.0], np.float32)
        # unit vector along horizontal rotation axis
        horizontal_axis = np.array([0.0, 0.0, 1.0], np.float32)
        # Rodrigues' rotation formula
        [R1, _] = cv2.Rodrigues(horizontal_axis * np.radians(THETA))
        [R2, _] = cv2.Rodrigues(np.dot(R1, vertical_axis) * np.radians(-PHI))

        # rotate the viewport
        xyz = xyz.reshape([height * width, 3]).T
        xyz = np.dot(R1, xyz)
        xyz = np.dot(R2, xyz).T
        # convert distance to latitude and longitude
        lat = np.arcsin(xyz[:, 2] / RADIUS)
        lon = np.zeros([height * width], np.float)
        theta = np.arctan(xyz[:, 1] / xyz[:, 0])
        
        # mask to crop out the subimages from equirectangular image
        idx1 = xyz[:, 0] > 0
        idx2 = xyz[:, 1] > 0
        idx3 = ((1 - idx1) * idx2).astype(np.bool)
        idx4 = ((1 - idx1) * (1 - idx2)).astype(np.bool)
        
        lon[idx1] = theta[idx1]
        lon[idx3] = theta[idx3] + np.pi
        lon[idx4] = theta[idx4] - np.pi
        # cooridinates of the mask at equirectangular image, in pixels
        lon = lon.reshape([height, width]) / np.pi * 180
        lat = -lat.reshape([height, width]) / np.pi * 180
        lon = lon / 180 * equ_cx + equ_cx
        lat = lat / 90 * equ_cy + equ_cy
        
        # sample equirectangular image based on coordinates
        persp = cv2.remap(self._img, lon.astype(np.float32), lat.astype(np.float32), cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP)
        return persp


def convert(theta, phi, save_path, filename, equ):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    img = equ.GetPerspective(104, theta, phi, 1440, 1600)
    start_saving = time.time()
    imsave(save_path + filename, img)
    # print("save image takes {}".format(time.time() - start_saving))

start = time.time()
count = 0
for root, dirs, files in os.walk("/usr/xtmp/ct214/daml/vr_sickness/pytorch-spynet/original-frames/skyhouse_frames/"):
    for file in files:
        if file.endswith("jpeg"): 
            count += 1
            path = os.path.join(root, file)
            if os.path.exists("/usr/xtmp/ct214/daml/vr_sickness/perspectives_skyhouse/left_eye/theta_-180_phi_-90.0/" + file):
                print("seen!", file)
                continue
            start_reading = time.time()
            input_img = imread(path)
            # print("imread takes {}".format(time.time() - start_reading))
            left_eye_img, right_eye_img = input_img[:input_img.shape[0]//2], input_img[input_img.shape[0]//2:]
            left_equ = Enquirec2Perspec(left_eye_img)
            right_equ = Enquirec2Perspec(right_eye_img) 
            print("start sliding ", file)
            for theta in range(-180, 180, 15):
                for phi in range(-90, 90, 15):
                    phi /= 2
                    start_converting = time.time()
                    convert(theta, phi, "/usr/xtmp/ct214/daml/vr_sickness/perspectives_skyhouse/left_eye/theta_" + str(theta) + "_phi_" + str(phi) + "/", file, left_equ)
                    # print("one convert takes {}".format(time.time() - start_converting))
                    convert(theta, phi, "/usr/xtmp/ct214/daml/vr_sickness/perspectives_skyhouse/right_eye/theta_" + str(theta) + "_phi_" + str(phi) + "/", file, right_equ)
            print("finish sliding ", file, " have seen ", str(count), " files already")
end = time.time()
print("===========================total time====================================")
print(- start + end)
