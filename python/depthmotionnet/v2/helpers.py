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

from depthmotionnet.helpers import *
 
def conv2d(inputs, num_outputs, kernel_size, data_format, **kwargs):
    """Convolution with 'same' padding"""

    return tf.layers.conv2d(
        inputs=inputs,
        filters=num_outputs,
        kernel_size=kernel_size,
        kernel_initializer=default_weights_initializer(),
        padding='same',
        data_format=data_format,
        **kwargs,
        )


def convrelu(inputs, num_outputs, kernel_size, data_format, **kwargs):
    """Shortcut for a single convolution+relu 
    
    See tf.layers.conv2d for a description of remaining parameters
    """
    return conv2d(inputs, num_outputs, kernel_size, data_format, activation=myLeakyRelu, **kwargs)


def convrelu2(inputs, num_outputs, kernel_size, name, stride, data_format, **kwargs):
    """Shortcut for two convolution+relu with 1D filter kernels 
    
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

    if isinstance(kernel_size,(tuple,list)):
        kernel_size_y = kernel_size[0]
        kernel_size_x = kernel_size[1]
    else:
        kernel_size_y = kernel_size
        kernel_size_x = kernel_size

    tmp_y = tf.layers.conv2d(
        inputs=inputs,
        filters=num_outputs_y,
        kernel_size=[kernel_size_y,1],
        strides=[stride,1],
        padding='same',
        activation=myLeakyRelu,
        kernel_initializer=default_weights_initializer(),
        data_format=data_format,
        name=name+'y',
        **kwargs,
    )
    return tf.layers.conv2d(
        inputs=tmp_y,
        filters=num_outputs_x,
        kernel_size=[1,kernel_size_x],
        strides=[1,stride],
        padding='same',
        activation=myLeakyRelu,
        kernel_initializer=default_weights_initializer(),
        data_format=data_format,
        name=name+'x',
        **kwargs,
    )


def recursive_median_downsample(inp, iterations):
    """Recursively downsamples the input using a 3x3 median filter"""
    result = []
    for i in range(iterations):
        if not result:
            tmp_inp = inp
        else:
            tmp_inp = result[-1]
        result.append(sops.median3x3_downsample(tmp_inp))
    return tuple(result)



