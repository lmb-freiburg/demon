#
#  DeMoN - Depth Motion Network
#  Copyright (C) 2017  Benjamin Ummenhofer, Huizhong Zhou
#  
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#  
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
import tensorflow as tf
import lmbspecialops as sops
import numpy as np
 
def convert_NCHW_to_NHWC(inp):
    """Convert the tensor from caffe format NCHW into tensorflow format NHWC
        
        inp: tensor 
    """
    return tf.transpose(inp,[0,2,3,1])

def convert_NHWC_to_NCHW(inp):
    """Convert the tensor from tensorflow format NHWC into caffe format NCHW 
        
        inp: tensor 
    """
    return tf.transpose(inp,[0,3,1,2])


def angleaxis_to_rotation_matrix(aa):
    """Converts the 3 element angle axis representation to a 3x3 rotation matrix
    
    aa: numpy.ndarray with 1 dimension and 3 elements

    Returns a 3x3 numpy.ndarray
    """
    angle = np.sqrt(aa.dot(aa))

    if angle > 1e-6:
        c = np.cos(angle);
        s = np.sin(angle);
        u = np.array([aa[0]/angle, aa[1]/angle, aa[2]/angle]);

        R = np.empty((3,3))
        R[0,0] = c+u[0]*u[0]*(1-c);      R[0,1] = u[0]*u[1]*(1-c)-u[2]*s; R[0,2] = u[0]*u[2]*(1-c)+u[1]*s;
        R[1,0] = u[1]*u[0]*(1-c)+u[2]*s; R[1,1] = c+u[1]*u[1]*(1-c);      R[1,2] = u[1]*u[2]*(1-c)-u[0]*s;
        R[2,0] = u[2]*u[0]*(1-c)-u[1]*s; R[2,1] = u[2]*u[1]*(1-c)+u[0]*s; R[2,2] = c+u[2]*u[2]*(1-c);
    else:
        R = np.eye(3)
    return R


def myLeakyRelu(x):
    """Leaky ReLU with leak factor 0.1"""
    # return tf.maximum(0.1*x,x)
    return sops.leaky_relu(x, leak=0.1)


def default_weights_initializer():
    return tf.contrib.layers.variance_scaling_initializer()


def conv2d_caffe_padding(inputs, num_outputs, kernel_size, data_format, **kwargs):
    """Convolution with 'same' padding as in caffe"""
    if isinstance(kernel_size,(tuple,list)):
        kernel_ysize = kernel_size[0]
        kernel_xsize = kernel_size[1]
    else:
        kernel_ysize = kernel_size
        kernel_xsize = kernel_size
    pad_y = kernel_ysize//2
    pad_x = kernel_xsize//2

    if data_format=='channels_first':
        paddings = [[0,0], [0,0], [pad_y, pad_y], [pad_x,pad_x]]
    else:
        paddings = [[0,0], [pad_y, pad_y], [pad_x,pad_x], [0,0]]
    padded_input = tf.pad(inputs, paddings=paddings)
    return tf.layers.conv2d(
        inputs=padded_input,
        filters=num_outputs,
        kernel_size=kernel_size,
        kernel_initializer=default_weights_initializer(),
        padding='valid',
        data_format=data_format,
        **kwargs,
        )


def convrelu_caffe_padding(inputs, num_outputs, kernel_size, data_format, **kwargs):
    """Shortcut for a single convolution+relu 
    
    See tf.layers.conv2d for a description of remaining parameters
    """
    return conv2d_caffe_padding(inputs, num_outputs, kernel_size, data_format, activation=myLeakyRelu, **kwargs)


def convrelu2_caffe_padding(inputs, num_outputs, kernel_size, name, stride, data_format, **kwargs):
    """Shortcut for two convolution+relu with 1D filter kernels and 'same' padding as in caffe
    
    num_outputs: int or (int,int)
        If num_outputs is a tuple then the first element is the number of
        outputs for the 1d filter in y direction and the second element is
        the final number of outputs.
    """
    if isinstance(num_outputs,(tuple,list)):
        num_outputs_y = num_outputs[0]
        num_outputs_x = num_outputs[1]
    else:
        num_outputs_y = num_outputs
        num_outputs_x = num_outputs

    pad = kernel_size//2

    if data_format=='channels_first':
        paddings_y = [[0,0], [0,0], [pad, pad], [0,0]]
        paddings_x = [[0,0], [0,0], [0,0], [pad, pad]]
    else:
        paddings_y = [[0,0], [pad, pad], [0,0], [0,0]]
        paddings_x = [[0,0], [0,0], [pad, pad], [0,0]]
    padded_input = tf.pad(inputs, paddings=paddings_y)

    tmp_y = tf.layers.conv2d(
        inputs=padded_input,
        filters=num_outputs_y,
        kernel_size=[kernel_size,1],
        strides=[stride,1],
        padding='valid',
        activation=myLeakyRelu,
        kernel_initializer=default_weights_initializer(),
        data_format=data_format,
        name=name+'y',
        **kwargs,
    )
    return tf.layers.conv2d(
        inputs=tf.pad(tmp_y, paddings=paddings_x),
        filters=num_outputs_x,
        kernel_size=[1,kernel_size],
        strides=[1,stride],
        padding='valid',
        activation=myLeakyRelu,
        kernel_initializer=default_weights_initializer(),
        data_format=data_format,
        name=name+'x',
        **kwargs,
    )

