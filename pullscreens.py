#pullGTAV Screens to Train
#todo--> add pytorch tensors in screen_grab
#todo--> have GTA V run on integrated graphics while using GPU for the model seperately
#todo --> change the Canny edge detection threshold based off of time of day in GTA V
#todo --> add store_data function to save screens for training later date
#todo --> move ML activities to train.py (image conversion to tensor) and use pullscreen for just processing
#todo --> Make screen_grab storage functions and create new screengrab that feeds as a stream

import sys
import time

from PIL import ImageGrab
import cv2
import numpy as np
import tensorflow as tf

flag_verbose = 0

#This is for POC --> use this approach which uses a
#CNN to process and pull edge data
#blog.altoros.com/using-convolutional-neural-networks-and-tensorflow-for-image-classification-and-search.html
def process_image_cv2(img):
    '''Pre process images w/ cv2 on cpu, converst to gray and uses canny to pull edges'''
    processed = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    processed = cv2.Canny(processed, threshold1 = 200, threshold2 = 300)
    return processed


def screen_grab(duration):
    '''Takes screenshots in PIL for a set duration and converts them to NP arrays'''
    start_time = time.time()
    last_time = time.time()
    first_loop = True
    duration += 1
    num_frames = 0

    while time.time()-start_time < duration:
        last_time = time.time()
        #change bbox per OS/Resolution/Scaling
        #bbox + 44 units for title bar from the corner of the screen
        screen = ImageGrab.grab(bbox=(0, 44, 800, 600))
        screen_np = np.array(screen)
        post_screen_np = process_image_cv2(screen_np)
        if flag_verbose and first_loop:
            #cv2.imsave('test.jpeg',image)
            cv2.imshow('NP Image', screen_np)
            cv2.imshow('NP Post', post_screen_np)
        if flag_verbose:
            print("Method took {} seconds".format(time.time()-last_time))
        first_loop = False
        num_frames += 1
        cv2.waitKey(25)


    print("Duration: {} Num Frames: {}".format(duration-1, num_frames))


def test_screen_grab_methods(lib, duration):
    '''Takes screengrab method and duration then tests
        IF multiple methods exist to find bottlenecks'''
    if lib == "cv2":
        screen_grab_cv2(duration, flag_verbose)
    if lib == "tf":
        screen_grab_tf(1, duration, flag_verbose)
        screen_grab_tf(2, duration, flag_verbose)

def screen_grab_tf(method, duration):
    '''Takes screenshots for a set duration in seconds, converts to tensor'''
    #init time
    start_time = time.time()
    last_time = time.time()
    first_loop = True
    #adds 1 duration as the initialization takes a second... this is ugly fix it later
    duration += 1
    num_frames = 0

    while time.time()-start_time < duration:
        #change bbox per OS/Resolution/Scaling
        #bbox + 44 units for title bar from the corner of the screen
        screen = ImageGrab.grab(bbox=(0, 44, 800, 600))

        if flag_verbose and first_loop:
            screen.show()

        #two methods (1) conver to tensor (2) tf.stack of np.array
        if(method == 1):
            #I think conver to tensor uses tf stack in implementation see documentation there
            tensored_screen = tf.convert_to_tensor(np.array(screen))
        else:
            tensored_screen = tf.stack(np.array(screen))


        if(flag_verbose): print("Method {}, took {} seconds".format(method, time.time()-last_time))
        first_loop = False
        num_frames += 1
        last_time = time.time()

    print("Method: {} Duration: {} Num Frames: {}".format(method, duration-1, num_frames))

def main():
    '''Main function, takes flag_verbose flag'''
    test_screen_grab_methods("cv2", 1)


if __name__ == "__main__":
    if sys.argv[1] == '-v':
        flag_verbose = 1
    main()
