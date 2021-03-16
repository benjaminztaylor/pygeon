import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.draw import random_shapes, rectangle
from skimage.util import img_as_ubyte
import pandas as pd
import datetime

# duration shape (or black) frame remains
frame_dur = 1 

# which shapes to 'choose' from
shape_arr = ['triangle', 'circle']

frame_size = (1080,1920)

# which shape is the conditioned stimulus

other_stim = 'circle'

cond_stim = 'triangle'

# randomly selects shapes (must be even!)
number_of_shapes = 10

shapes = np.random.choice(shape_arr, number_of_shapes, p=[0.5,0.5])

# This creates a blank image with the correct dims
h, w = frame_size

time_key = pd.DataFrame(shapes, columns=['stimulus_shape'])
time_key['cameras'] = np.tile((1,2),int(number_of_shapes/2))
time_key['duration'] = np.nan
time_key.loc[time_key['stimulus_shape'] == cond_stim, 'duration'] = 21
time_key.loc[time_key['stimulus_shape'] == other_stim, 'duration'] = 5

# shift since first frame begins with fixation pt
time_key.at[0,'duration'] = time_key.at[0,'duration'] + 16

time_key['time'] = time_key['duration'].cumsum()
time_key['times'] = pd.to_timedelta(time_key['time'], unit='m',)

# Save times
today = datetime.date.today()
today = today.strftime("%d-%m-%Y")
time_key_df_name = '{}_{}'.format(today, 'time_key.csv')
time_key.to_csv(time_key_df_name)


fix_img = np.full((h, w), 255, dtype = np.uint8)
start = (540,960)
extent = (590,1010)
rr, cc = rectangle(start, end=extent, shape=fix_img.shape)
fix_img[rr, cc] = 0


#each holds camera images 
cam1_images = []
cam2_images = []
cam3_images = []

h, w = frame_size
blank_img = np.full((h, w), 255, dtype = np.uint8)

def cam3_fixation():
    ''' Generates 16 images with the first image as the fixation image.  
    '''
    cam3_images.append(fix_img)
    for _ in range(15):
        cam3_images.append(blank_img)

def append_single(cam, num):
    ''' for appending blank images to a single camera.
            cam = name of camera image folder;
            num = number of blank frames to append
    '''
    for _ in range(num):
        cam.append(blank_img)

def append_multiple(first_cam, second_cam, num):
    ''' for appending blank images to a single camera.
            first_cam = name of camera image folder;
            second_cam = name of other camera image folder;
            num = number of blank frames to append
    '''
    for _ in range(num):
        first_cam.append(blank_img)
        second_cam.append(blank_img)

def get_results(shapes,camera_images):
    result = random_shapes((120,1920), intensity_range=(0, 1), max_shapes=1, min_size = 99, max_size=100, shape=shapes, multichannel=False)
    result_pad = np.pad(result[0], ((480,480),(0,0)), mode='constant', constant_values=255)
    camera_images.append(result_pad)
    append_single(camera_images, 4)


# Generates first frame

cam3_fixation()
append_multiple(cam1_images, cam2_images, 16)


for cam1_shapes, cam2_shapes in zip(shapes, shapes[1:]):
    # if camera 1 is condition stimulus shape 
    if (cam1_shapes == cond_stim) & (cam2_shapes != cond_stim):
        get_results(cam1_shapes,cam1_images)

        # camera 2 and 3
        append_multiple(cam2_images,cam3_images,5)
        # camera 3 only
        cam3_fixation()
        # for camera 1 and 2 (delay from fixation point)
        append_multiple(cam1_images,cam2_images,16)

        # write camera2 after
        get_results(cam2_shapes, cam2_images)
        # camera 1 and 3 (blank images)
        append_multiple(cam1_images,cam3_images,5)
    # if camera 2 is condition stimulus shape
    elif (cam2_shapes == cond_stim) & (cam1_shapes != cond_stim):
        # write camera 1 first
        get_results(cam1_shapes,cam1_images)
        append_multiple(cam2_images,cam3_images,5)
        # camera 1 and 3 (blank images)
        get_results(cam2_shapes,cam2_images)
        append_multiple(cam1_images,cam3_images,5)
        # camera 3 only
        cam3_fixation()
        # for camera 1 and 2 (delay from fixation point)
        append_multiple(cam1_images, cam2_images, 16)

    elif (cam2_shapes != cond_stim) & (cam1_shapes != cond_stim):
        cam_array = ['cam1', 'cam2']
        which_array = np.random.choice(cam_array, 1, p=[0.5,0.5])

        if which_array == 'cam1':
            get_results(cam1_shapes,cam1_images)
            append_multiple(cam2_images, cam3_images, 5)

            get_results(cam2_shapes,cam2_images)
            append_multiple(cam1_images, cam3_images, 5)

        else:
            get_results(cam2_shapes,cam2_images)
            append_multiple(cam1_images, cam3_images, 5)

            get_results(cam1_shapes,cam1_images)
            append_multiple(cam2_images, cam3_images, 5)

# if both are condition shapes select randomly which frame goes first
    elif (cam2_shapes == cond_stim) & (cam1_shapes == cond_stim):
        cam_array = ['cam1', 'cam2']
        which_array = np.random.choice(cam_array, 1, p=[0.5,0.5])

        if which_array == 'cam1':
            get_results(cam1_shapes,cam1_images)
            append_multiple(cam2_images, cam3_images, 5)
            # camera 3 fixation
            cam3_fixation()

    # for camera 1 and 2 (delay from fixation point)
            append_multiple(cam1_images, cam2_images, 16)
            get_results(cam2_shapes,cam2_images)
            append_multiple(cam1_images, cam3_images, 5)
            # camera 3 only
            cam3_fixation()
            append_multiple(cam1_images, cam2_images, 16)
        
        else:
            get_results(cam2_shapes,cam2_images)
            append_multiple(cam1_images, cam3_images, 5)
            # camera 3 only
            cam3_fixation()
            # for camera 1 and 2 (delay from fixation point)
            append_multiple(cam1_images, cam2_images, 16)
            get_results(cam1_shapes,cam1_images)
            append_multiple(cam2_images, cam3_images, 5)
            # camera 3 only
            cam3_fixation()
            append_multiple(cam1_images, cam2_images, 16)
            # for camera 1 and 2 (delay from fixation point)


# writing videos

video_name = '{}{}'.format('camera_01_test', '.mp4' )
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

video = cv2.VideoWriter(video_name, fourcc, 1 , (w,h), 0)

for frame in cam1_images:
    frame = img_as_ubyte(frame)
    video.write(frame)
video.release()

video_name = '{}{}'.format('camera_02_test', '.mp4' )



video = cv2.VideoWriter(video_name, fourcc, 1 , (w,h), 0)

for frame in cam2_images:
    frame = img_as_ubyte(frame)
    video.write(frame)
video.release()


video_name = '{}{}'.format('camera_03_test', '.mp4' )

video = cv2.VideoWriter(video_name, fourcc, 1 , (w,h), 0)

for frame in cam3_images:
    frame = img_as_ubyte(frame)
    video.write(frame)
video.release()