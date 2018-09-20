import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt 
import argparse as arg
import os 
import cv2
import glob2



## prepare the data -- input the images
## process these images, at least normalize them to [-1,1]


'''
Notice that the import is nothing,i.e.directly create the output from the conditioning input
"We do not inject a noise vector while training our network to produce deterministic outputs"
the condition input is W*H*3Ch*3Ph*Nw=11 -- current frame at its end. 
'''

## the Generator: reference to the paper.
## the Discriminator: patchGAN + body2body. the real_in is the current frame and compare it to the G one.
## notice there is an another loss function which match the extent of similarity between y(true) and G(x) (x:condition)
