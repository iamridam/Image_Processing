from PIL import Image
import cv2
import numpy as np

img = np.zeros([1485,1016,3])

scene_infile = open('./1.raw','rb')
scene_image_array = np.fromfile(scene_infile,dtype=np.uint8,count=1016*1485)

scene_image = Image.frombuffer("I",[1016,1485],
                                 scene_image_array.astype('I'),
                                 'raw','I', 0, 1)
scene_image = np.array(scene_image)

img[:,:,2] = scene_image

scene_infile = open('./2.raw','rb')
scene_image_array = np.fromfile(scene_infile,dtype=np.uint8,count=1016*1485)

scene_image = Image.frombuffer("I",[1016,1485],
                                 scene_image_array.astype('I'),
                                 'raw','I', 0, 1)
scene_image = np.array(scene_image)

img[:,:,1] = scene_image

scene_infile = open('./3.raw','rb')
scene_image_array = np.fromfile(scene_infile,dtype=np.uint8,count=1016*1485)

scene_image = Image.frombuffer("I",[1016,1485],
                                 scene_image_array.astype('I'),
                                 'raw','I', 0, 1)
scene_image = np.array(scene_image)

img[:,:,0] = scene_image




cv2.imwrite('test3.jpg', img)