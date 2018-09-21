## this is the structure of the net
'''
在生成器和判别器中，image参数就是指的条件
并且在生成器的输入中，随机噪声被去掉了(仅仅输入了条件)
在判别器的输入中，条件和待判别的图像被拼接(concat)了起来

https://blog.csdn.net/jiongnima/article/details/80209239?utm_source=copy 
'''
import numpy as np
import tensorflow as tf
import math

# define convolution layer
# with kernel_size = 4 and stride = 2 - down
# with kernel_size = 3 and stride = 1 - up
def conv2d(input_, output_dim, kernel_size, stride, padding = "SAME", scope_name = "conv2d", biased = False):
    input_chan = input_.get_shape()[-1]
    with tf.variable_scope(scope_name):
        kernel = tf.get_variable(name = 'weights', shape=[kernel_size, kernel_size, input_chan, output_dim], trainable=True)
        output = tf.nn.conv2d(input_, kernel, [1, stride, stride, 1], padding = padding)
        if biased:
            biases = tf.get_variable(name = 'biases', shape = [output_dim], trainable=True)
            output = tf.nn.bias_add(output, biases)
        return output

#定义空洞卷积层
def atrous_conv2d(input_, output_dim, kernel_size, dilation, padding = "SAME", scope_name = "atrous_conv2d", biased = False):
    input_chan = input_.get_shape()[-1]
    with tf.variable_scope(scope_name):
        kernel = tf.get_variable(name = 'weights', shape = [kernel_size, kernel_size, input_chan, output_dim], trainable=True)
        output = tf.nn.atrous_conv2d(input_, kernel, dilation, padding = padding)
        if biased:
            biases = tf.get_variable(name = 'biases', shape = [output_dim], trainable=True)
            output = tf.nn.bias_add(output, biases)
        return output

#定义反卷积层
def deconv2d(input_, output_dim, kernel_size, stride, padding = "SAME", scope_name = "deconv2d"):
    input_chan = input_.get_shape()[-1]
    input_height = int(input_.get_shape()[1])
    input_width = int(input_.get_shape()[2])
    with tf.variable_scope(scope_name):
        kernel = tf.get_variable(name = 'weights', shape = [kernel_size, kernel_size, output_dim, input_chan], trainable=True)
        output = tf.nn.conv2d_transpose(input_, kernel, [1, input_height * 2, input_width * 2, output_dim], [1, 2, 2, 1], padding = "SAME")
        return output

#define batch normalization(BN) layer
def batch_norm(input_, scope_name="batch_norm"):
    with tf.variable_scope(scope_name):
        input_chan = input_.get_shape()[-1]
        scale = tf.get_variable("scale", [input_chan], initializer=tf.random_normal_initializer(1.0, 0.02, dtype=tf.float32))
        offset = tf.get_variable("offset", [input_chan], initializer=tf.constant_initializer(0.0))
        mean, variance = tf.nn.moments(input_, axes=[1,2], keep_dims=True)
        epsilon = 1e-5
        inv = tf.rsqrt(variance + epsilon)
        normalized = (input_-mean)*inv
        output = scale*normalized + offset
        return output

# lrelu activation
def lrelu(x, leak=0.2, scope_name = "lrelu"):
    return tf.maximum(x, leak*x)

#generator + UNet
def generator(image, base_dim=64, reuse=False, scope_name="generator"):
    # input dimension is [1,256,256,11*3*3]
    input_chan = int(image.get_shape()[-1]) # TODO = 99?
    dropout_rate = [0.5,0.5,0.5,0,0,0,0,0] #定义dropout的比例
    with tf.variable_scope(scope_name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

	    #the first convolution layer output: [1, 128, 128, 64]
        c1 = conv2d(input_=image, output_dim=base_dim, kernel_size=4, stride=2, scope_name='g_c1_conv')
        #lrelu the outcome
        c1_lr = lrelu(c1)

	    #the second convolution layer output: [1, 64, 64, 128]
        c2 = conv2d(input_=c1_lr, output_dim=base_dim*2, kernel_size=4, stride=2, scope_name='g_c2_conv')
        #batch normalization c2
        c2_bn = batch_norm(c2, scope_name='g_bn_c2')
        #lrelu the outcome
        c2_bn_lr = lrelu(c2_bn)

        #the third convolution layer output: [1, 32, 32, 256]
        c3 = conv2d(input_=c2_bn_lr, output_dim=base_dim*4, kernel_size=4, stride=2, scope_name='g_c3_conv')
        #batch normalization c3
        c3_bn = batch_norm(c3, scope_name='g_bn_c3')
        #lrelu the outcome
        c3_bn_lr = lrelu(c3_bn)
    
        #the forth convolution layer output: [1, 16, 16, 512]
        c4 = conv2d(input_=c3_bn_lr, output_dim=base_dim*8, kernel_size=4, stride=2, scope_name='g_c4_conv')
        #batch normalization c4
        c4_bn = batch_norm(c4, scope_name='g_bn_c4')
        #lrelu the outcome
        c4_bn_lr = lrelu(c4_bn)

        #the forth convolution layer output: [1, 8, 8, 512]
        c5 = conv2d(input_=c4_bn_lr, output_dim=base_dim*8, kernel_size=4, stride=2, scope_name='g_c5_conv')
        #batch normalization c5
        c5_bn = batch_norm(c5, scope_name='g_bn_c5')
        #lrelu the outcome
        c5_bn_lr = lrelu(c5_bn)

        #the forth convolution layer output: [1, 4, 4, 512]
        c6 = conv2d(input_=c5_bn_lr, output_dim=base_dim*8, kernel_size=4, stride=2, scope_name='g_c6_conv')
        #batch normalization c6
        c6_bn = batch_norm(c6, scope_name='g_bn_c6')
        #lrelu the outcome
        c6_bn_lr = lrelu(c6_bn)

        #the forth convolution layer output: [1, 2, 2, 512]
        c7 = conv2d(input_=c6_bn_lr, output_dim=base_dim*8, kernel_size=4, stride=2, scope_name='g_c7_conv')
        #batch normalization c7
        c7_bn = batch_norm(c7, scope_name='g_bn_c7')
        #lrelu the outcome
        c7_bn_lr = lrelu(c7_bn)

        #the forth convolution layer output: [1, 1, 1, 512]
        c8 = conv2d(input_=c7_bn_lr, output_dim=base_dim*8, kernel_size=4, stride=2, scope_name='g_c8_conv')
        #batch normalization c8
        c8_bn = batch_norm(c8, scope_name='g_bn_c8')
        #lrelu the outcome
        c8_bn_lr = lrelu(c8_bn)

	    #the first deconvolution layer output:[1, 2, 2, 512]
        d1 = deconv2d(input_=c8_bn_lr, output_dim=base_dim*8, kernel_size=4, stride=2, scope_name='g_d1')
        #batch normalization d1
        d1_bn = batch_norm(d1, scope_name='g_bn_d1')
        #random dropout 0.5
        d1_bn_dr = tf.nn.dropout(d1_bn, dropout_rate[0])
        #relu
        d1_bn_dr_rl = tf.nn.relu(d1_bn_dr)
        #d1_refine layer output:[1, 2, 2, 512]
        d1_rf1 = conv2d(input_=d1_bn_dr_rl, output_dim=base_dim*8, kernel_size=3, stride=1,scope_name='g_d1_rf1')
        #batch normalization d1_rf1
        d1_rf1_bn = batch_norm(d1_rf1, scope_name='g_bn_d1_rf1')
        #random dropout 0.5
        d1_rf1_bn_dr = tf.nn.dropout(d1_rf1_bn, dropout_rate[0])
        #relu
        d1_rf1_bn_dr_rl = tf.nn.relu(d1_rf1_bn_dr)
        #d1_refine layer output:[1, 2, 2, 512]
        d1_rf2 = conv2d(input_=d1_rf1_bn_dr_rl, output_dim=base_dim*8, kernel_size=3, stride=1,scope_name='g_d1_rf2')
        #batch normalization d1_rf2
        d1_rf2_bn = batch_norm(d1_rf2, scope_name='g_bn_d1_rf2')
        #random dropout 0.5
        d1_rf2_bn_dr = tf.nn.dropout(d1_rf2_bn, dropout_rate[0])
        #relu
        d1_rf2_bn_dr_rl = tf.nn.relu(d1_rf2_bn_dr)
        
	    #the second deconvolution layer output:[1, 4, 4, 512]
        d2 = deconv2d(input_=d1_rf2_bn_dr_rl, output_dim=base_dim*8, kernel_size=4, stride=2, scope_name='g_d2')
        #batch normalization d2
        d2_bn = batch_norm(d2, scope_name='g_bn_d2')
        #random dropout 0.5
        d2_bn_dr = tf.nn.dropout(d2_bn, dropout_rate[1])
        #relu
        d2_bn_dr_rl = tf.nn.relu(d2_bn_dr)
        #d2_refine layer output:[1, 4, 4, 512]
        d2_rf1 = conv2d(input_=d2_bn_dr_rl, output_dim=base_dim*8, kernel_size=3, stride=1,scope_name='g_d2_rf1')
        #batch normalization d2_rf1
        d2_rf1_bn = batch_norm(d2_rf1, scope_name='g_bn_d2_rf1')
        #random dropout 0.5
        d2_rf1_bn_dr = tf.nn.dropout(d2_rf1_bn, dropout_rate[1])
        #relu
        d2_rf1_bn_dr_rl = tf.nn.relu(d2_rf1_bn_dr)
        #d2_refine layer output:[1, 4, 4, 512]
        d2_rf2 = conv2d(input_=d2_rf1_bn_dr_rl, output_dim=base_dim*8, kernel_size=3, stride=1,scope_name='g_d2_rf2')
        #batch normalization d2_rf2
        d2_rf2_bn = batch_norm(d2_rf2, scope_name='g_bn_d2_rf2')
        #random dropout 0.5
        d2_rf2_bn_dr = tf.nn.dropout(d2_rf2_bn, dropout_rate[1])
        #relu
        d2_rf2_bn_dr_rl = tf.nn.relu(d2_rf2_bn_dr)

        #TODO next:




        d1 = tf.concat([batch_norm(d1, scope_name='g_bn_d1'), e7], 3)
	#第二个反卷积层，输出尺度[1, 4, 4, 512]
        d2 = deconv2d(input_=tf.nn.relu(d1), output_dim=base_dim*8, kernel_size=4, stride=2, scope_name='g_d2')
        d2 = tf.nn.dropout(d2, dropout_rate) #随机扔掉一般的输出
        d2 = tf.concat([batch_norm(d2, scope_name='g_bn_d2'), e6], 3)
	#第三个反卷积层，输出尺度[1, 8, 8, 512]
        d3 = deconv2d(input_=tf.nn.relu(d2), output_dim=base_dim*8, kernel_size=4, stride=2, scope_name='g_d3')
        d3 = tf.nn.dropout(d3, dropout_rate) #随机扔掉一般的输出
        d3 = tf.concat([batch_norm(d3, scope_name='g_bn_d3'), e5], 3)
	#第四个反卷积层，输出尺度[1, 16, 16, 512]
        d4 = deconv2d(input_=tf.nn.relu(d3), output_dim=base_dim*8, kernel_size=4, stride=2, scope_name='g_d4')
        d4 = tf.concat([batch_norm(d4, scope_name='g_bn_d4'), e4], 3)
	#第五个反卷积层，输出尺度[1, 32, 32, 256]
        d5 = deconv2d(input_=tf.nn.relu(d4), output_dim=base_dim*4, kernel_size=4, stride=2, scope_name='g_d5')
        d5 = tf.concat([batch_norm(d5, scope_name='g_bn_d5'), e3], 3)
	#第六个反卷积层，输出尺度[1, 64, 64, 128]
        d6 = deconv2d(input_=tf.nn.relu(d5), output_dim=base_dim*2, kernel_size=4, stride=2, scope_name='g_d6')
        d6 = tf.concat([batch_norm(d6, scope_name='g_bn_d6'), e2], 3)
	#第七个反卷积层，输出尺度[1, 128, 128, 64]
        d7 = deconv2d(input_=tf.nn.relu(d6), output_dim=base_dim, kernel_size=4, stride=2, scope_name='g_d7')
        d7 = tf.concat([batch_norm(d7, scope_name='g_bn_d7'), e1], 3)
	#第八个反卷积层，输出尺度[1, 256, 256, 3]
        d8 = deconv2d(input_=tf.nn.relu(d7), output_dim=input_chan, kernel_size=4, stride=2, scope_name='g_d8')
        return tf.nn.tanh(d8)

#定义判别器
def discriminator(image, targets, df_dim=64, reuse=False, scope_name="discriminator"):
    with tf.variable_scope(scope_name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False
        dis_input = tf.concat([image, targets], 3)
	#第1个卷积模块，输出尺度: 1*128*128*64
        h0 = lrelu(conv2d(input_ = dis_input, output_dim = df_dim, kernel_size = 4, stride = 2, scope_name='d_h0_conv'))
	#第2个卷积模块，输出尺度: 1*64*64*128
        h1 = lrelu(batch_norm(conv2d(input_ = h0, output_dim = df_dim*2, kernel_size = 4, stride = 2, scope_name='d_h1_conv'), scope_name='d_bn1'))
	#第3个卷积模块，输出尺度: 1*32*32*256
        h2 = lrelu(batch_norm(conv2d(input_ = h1, output_dim = df_dim*4, kernel_size = 4, stride = 2, scope_name='d_h2_conv'), scope_name='d_bn2'))
	#第4个卷积模块，输出尺度: 1*32*32*512
        h3 = lrelu(batch_norm(conv2d(input_ = h2, output_dim = df_dim*8, kernel_size = 4, stride = 1, scope_name='d_h3_conv'), scope_name='d_bn3'))
	#最后一个卷积模块，输出尺度: 1*32*32*1
        output = conv2d(input_ = h3, output_dim = 1, kernel_size = 4, stride = 1, scope_name='d_h4_conv')
        dis_out = tf.sigmoid(output) #在输出之前经过sigmoid层，因为需要进行log运算
        return dis_out