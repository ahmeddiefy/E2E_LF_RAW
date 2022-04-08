import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

kernel_size = 3
f = 64
n_res_blocks=5
n_res_groups=3
n_res_interpolation=1
spatial_f = 4
weights = []
w_init = tf.variance_scaling_initializer(scale=1., mode='fan_avg', distribution="uniform")
b_init = tf.zeros_initializer()






def conv2d(x, f_in, f_out, k, name):
    """
    :param x: input
    :param f: filters
    :param k: kernel size
    :param s: strides
    :param pad: padding
    :param use_bias: using bias or not
    :param reuse: reusable
    :param name: scope name
    :return: output
    """
    
    conv_w = tf.get_variable(name + "_w" , [k,k,f_in,f_out], initializer=w_init)
    conv_b = tf.get_variable(name + "_b" , [f_out], initializer=b_init)
    weights.append(conv_w)
    weights.append(conv_b)
    return tf.nn.bias_add(tf.nn.conv2d(x, conv_w, strides=[1,1,1,1], padding='SAME'), conv_b)                           


def spatial_residual_block(x, name):
    skip_conn = x  
    conv_name = "conv2d-1" + "_" + name
    x = conv2d(x, f_in=f*spatial_f, f_out=f*spatial_f, k=kernel_size, name=conv_name)
    x = tf.nn.relu(x)
    conv_name = "conv2d-2" + "_" + name
    x = conv2d(x, f_in=f*spatial_f, f_out=f*spatial_f, k=kernel_size, name=conv_name)
    x = tf.nn.relu(x)
    conv_name = "conv2d-3" + "_" + name
    x = conv2d(x, f_in=f*spatial_f, f_out=f*spatial_f, k=kernel_size, name=conv_name)
    

    return tf.add(x , skip_conn)            
            
            
def residual_block(x, name):
    skip_conn = x  
    conv_name = "conv2d-1" + "_" + name
    x = conv2d(x, f_in=f, f_out=f, k=kernel_size, name=conv_name)
    x = tf.nn.relu(x)
    conv_name = "conv2d-2" + "_" + name
    x = conv2d(x, f_in=f, f_out=f, k=kernel_size, name=conv_name)
    x = tf.nn.relu(x)
    conv_name = "conv2d-3" + "_" + name
    x = conv2d(x, f_in=f, f_out=f, k=kernel_size, name=conv_name)

    return tf.add(x , skip_conn)
    
def spatial_residual_group(x, name):
    conv_name = "conv2d-1" + "_" + name
    x = conv2d(x, f_in=f*spatial_f, f_out=f*spatial_f, k=kernel_size, name=conv_name)
    skip_conn = x

    for i in range(n_res_blocks):
        x = spatial_residual_block(x, name=name + "_" + str(i))
        
    conv_name = "rg-conv-" + name
    x = conv2d(x, f_in=f*spatial_f, f_out=f*spatial_f, k=kernel_size, name=conv_name)
    return tf.add(x , skip_conn)    
    
    
def residual_group(x, name):
    conv_name = "conv2d-1" + "_" + name
    x = conv2d(x, f_in=f, f_out=f, k=kernel_size, name=conv_name)
    skip_conn = x

    for i in range(n_res_blocks):
        x = residual_block(x, name=name + "_" + str(i))
        
    conv_name = "rg-conv-" + name
    x = conv2d(x, f_in=f, f_out=f, k=kernel_size, name=conv_name)
    return tf.add(x , skip_conn)
            
            
def spatial_augmentation_network(x, count_name):
    # 1. head
    x = conv2d(x, f_in=f, f_out=f*spatial_f, k=kernel_size, name="conv2d-head"+count_name)
    head = x
    # 2. body

    for i in range(n_res_groups):
        x = spatial_residual_group(x, name=str(i) + count_name)

    body = conv2d(x, f_in=f*spatial_f, f_out=f*spatial_f, k=kernel_size, name="conv2d-body"+count_name)
    body = tf.nn.relu(body)

    tail = conv2d(body, f_in=f*spatial_f, f_out=f, k=kernel_size, name="conv2d-tail"+count_name)  

    return tail            
            
            

            
def angular_augmentation_network(x, count_name):
    # 1. head
    x = conv2d(x, f_in=1, f_out=f, k=kernel_size, name="conv2d-head"+count_name)
    head = x
    # 2. body

    for i in range(n_res_groups):
        x = residual_group(x, name=str(i) + count_name)

    body = conv2d(x, f_in=f, f_out=f, k=kernel_size, name="conv2d-body"+count_name)
    body = tf.nn.relu(body)

    tail = conv2d(body, f_in=f, f_out=1, k=kernel_size, name="conv2d-tail"+count_name)  

    return tail
    
                
def residual_interpolation_network(x):
    # 1. head
    x = conv2d(x, f_in=1, f_out=f, k=kernel_size, name="conv2d-head_interpolation")
    head = x

    # 2. body

    for i in range(n_res_interpolation):
        x = residual_group(x, name="interpolation-" + str(i) )

    body = conv2d(x, f_in=f, f_out=f, k=kernel_size, name="conv2d-body_interpolation")
    body = tf.nn.relu(body)

    tail = conv2d(body, f_in=f, f_out=16, k=kernel_size, name="conv2d-tail_interpolation")  

    return tail

      
            
def model(input_tensor):
    with tf.device("/gpu:0"):
        tensor = None
        upscaled_tensor = None
        
        print(input_tensor.shape)
        tensor = tf.depth_to_space(input_tensor, block_size=2)
        print(tensor.shape)
        
        tensor = residual_interpolation_network(tensor)
        print(tensor.shape)
        
        tensor = tf.depth_to_space(tensor, block_size=4)
        print(tensor.shape)
        tensor = tf.space_to_depth(tensor, block_size=8)
        print(tensor.shape)
        tensor = tf.concat( [tensor[:,:,:,0:9], tf.expand_dims(input_tensor[:,:,:,0],axis=3), tensor[:,:,:,10:14], tf.expand_dims(input_tensor[:,:,:,1],axis=3), tensor[:,:,:,15:49], tf.expand_dims(input_tensor[:,:,:,2],axis=3), tensor[:,:,:,50:54], tf.expand_dims(input_tensor[:,:,:,3],axis=3), tensor[:,:,:,55:64],], axis=3)
        print(tensor.shape)
        
        upscaled_tensor = tf.depth_to_space(tensor, block_size=8)
        
        print(tensor.shape)
                             
        tensor = tf.depth_to_space(tensor, block_size=8)
        print(tensor.shape)
			
        tensor = angular_augmentation_network(tensor, "Angular_first")
        print(tensor.shape)

               
        tensor = tf.add(tensor, upscaled_tensor)
 
        print(tensor.shape)        
        
        
        return tensor, weights
		 