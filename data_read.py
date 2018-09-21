## data preparation
## notice in the future we should load at least 11 photos

import os
import numpy as np
import tensorflow as tf
import cv2

#读取图片的函数，接收六个参数
#输入参数分别是图片名，图片path，标签path，图片格式，标签格式，需要调整的尺寸大小
def ImageReader(file_name, picture_path, label_path, picture_format = ".png", label_format = ".jpg", size = 256):
    picture_name = picture_path + file_name + picture_format #得到图片名称和路径
    label_name = label_path + file_name + label_format #得到标签名称和路径
    picture = cv2.imread(picture_name, 1) #读取图片
    label = cv2.imread(label_name, 1) #读取标签
    height = picture.shape[0] #得到图片的高
    width = picture.shape[1] #得到图片的宽
    picture_resize_t = cv2.resize(picture, (size, size)) #调整图片的尺寸，改变成网络输入的大小
    picture_resize = picture_resize_t / 127.5 - 1. #归一化图片
    label_resize_t = cv2.resize(label, (size, size)) #调整标签的尺寸，改变成网络输入的大小
    label_resize = label_resize_t / 127.5 - 1. #归一化标签
    return picture_resize, label_resize, height, width #返回网络输入的图片，标签，还有原图片和标签的长宽 

#TODO map the input to RGB [-1,1]