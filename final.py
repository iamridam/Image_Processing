import sys
import numpy as np
from PIL import Image
import itertools
import math

def mean_fun(pixel_sum,class_count): # function for calculating mean of each class
    i = 0
    mean_arr = [0] * 14
    for x, y in itertools.izip(pixel_sum, class_count):
        mean_arr[i] = round(x / y, 2)
        i += 1
    return mean_arr

def variance_fun(class_list,mean,class_count):
    variance = [0]*14
    vsum = 0
    for x in range(14):
        for y in class_list[x]:
            d = mean[x] - y
            if d < 0:
                d = -d
            p=d*d
            vsum += p
        variance[x] = round(vsum/class_count[x],2)
        vsum = 0
    return variance

def gauss_fun(mean,variance,pixel_value):
    d = (pixel_value-mean)/variance
    ln = math.exp(-0.5*(d**2))*10000
    deno = 2.5066*variance
    return ln/deno

img = np.zeros([1485,1016,2])

scene_infile = open('Train1016x1485.raw', 'rb')
scene_image_array = np.fromfile(scene_infile, dtype=np.uint8, count=1016 * 1485)
scene_image = Image.frombuffer("I", [1016, 1485], scene_image_array.astype('I'), 'raw', 'I', 0, 1)
scene_image = np.array(scene_image)
img[:, :, 0] = scene_image

scene_infile = open('5.raw','rb')
scene_image_array = np.fromfile(scene_infile,dtype=np.uint8,count=1016*1485)
scene_image = Image.frombuffer("I",[1016,1485],scene_image_array.astype('I'),'raw','I', 0, 1)
scene_image = np.array(scene_image)
img[:,:,1] = scene_image

test_img = np.zeros([1485,1016])
scene_infile = open('Test1016x1485.raw','rb')
scene_image_array = np.fromfile(scene_infile,dtype=np.uint8,count=1016*1485)
scene_image = Image.frombuffer("I",[1016,1485],scene_image_array.astype('I'),'raw','I', 0, 1)
scene_image = np.array(scene_image)
test_img[:,:] = scene_image

pixel_sum=[0]*14 # No of pixels of each class in radar image
counter = 0
class_count = [0]*14 # Total number of classes of each category in training image
class_list = [[] for _ in range(14)] # list containing pixel values of each class

for x in img:
    for y in x:
        for z in y:
            if counter == 0:
                class_val = int(z)
                counter += 1
            else:
                counter -= 1
                if class_val != 0:
                    pixel_sum[class_val-1]+= z
                    class_count[class_val-1]+= 1
                    class_list[class_val-1].append(z)

mean = mean_fun(pixel_sum,class_count)
variance = variance_fun(class_list,mean,class_count)

resultant = np.zeros([1485,1016])
highest_pixel_value = -sys.maxint-1

for x in range(1485):
    for y in range(1016):
        pixel_value = img[x][y][1]
        #class_value = img[x][y][0]
        class_value = test_img[x][y]
        if class_value != 0:
            for z in range(14):
                fun_val = gauss_fun(mean[z],variance[z],pixel_value)
                if fun_val < 0:
                    fun_val = -fun_val
                if fun_val>highest_pixel_value:
                    class_value = z+1
                    highest_pixel_value = fun_val

        resultant[x][y] = class_value
        highest_pixel_value = -sys.maxint - 1

match = 0

for x in range(1485):
    for y in range(1016):
        if(test_img[x][y] != 0 and (resultant[x][y]== test_img[x][y])):
        #if (img[x][y][0] != 0 and (resultant[x][y] == img[x][y][0])):
            match+=1

'''for x in range(5):
    for y in range(5):
        print "test  ",test_img[x][y],"   res   ",resultant[x][y]'''

total_pixel = sum(pixel_sum[1:14])
print "accuracy = ", match/total_pixel
print match
print total_pixel
