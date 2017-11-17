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
import os, sys
import tensorflow as tf
from .helpers import *
import lmbspecialops as sops


def _predict_flow(inp, predict_confidence=False, **kwargs ):
    """Generates a tensor for optical flow prediction
    
    inp: Tensor

    predict_confidence: bool
        If True the output tensor has 4 channels instead of 2.
        The last two channels are the x and y flow confidence.
    """

    tmp = convrelu(
        inputs=inp,
        num_outputs=24,
        kernel_size=3,
        strides=1,
        name="conv1",
        **kwargs,
    )
    
    output = conv2d(
        inputs=tmp,
        num_outputs=4 if predict_confidence else 2,
        kernel_size=3,
        strides=1,
        name="conv2",
        **kwargs,
    )
    
    return output


def _upsample_prediction(inp, num_outputs, **kwargs ):
    """Upconvolution for upsampling predictions
    
    inp: Tensor 
        Tensor with the prediction
        
    num_outputs: int
        Number of output channels. 
        Usually this should match the number of channels in the predictions
    """
    output = tf.layers.conv2d_transpose(
        inputs=inp,
        filters=num_outputs,
        kernel_size=4,
        strides=2,
        padding='same',
        activation=None,
        kernel_initializer=default_weights_initializer(),
        name="upconv",
        **kwargs,
    )
    return output



def _refine(inp, num_outputs, data_format, upsampled_prediction=None, features_direct=None, **kwargs):
    """ Generates the concatenation of 
         - the previous features used to compute the flow/depth
         - the upsampled previous flow/depth
         - the direct features that already have the correct resolution

    inp: Tensor
        The features that have been used before to compute flow/depth

    num_outputs: int 
        number of outputs for the upconvolution of 'features'

    upsampled_prediction: Tensor
        The upsampled flow/depth prediction

    features_direct: Tensor
        The direct features which already have the spatial output resolution
    """
    upsampled_features = tf.layers.conv2d_transpose(
        inputs=inp,
        filters=num_outputs,
        kernel_size=4,
        strides=2,
        padding='same',
        activation=myLeakyRelu,
        kernel_initializer=default_weights_initializer(),
        data_format=data_format,
        name="upconv",
        **kwargs,
    )
    inputs = [upsampled_features, features_direct, upsampled_prediction]
    concat_inputs = [ x for x in inputs if not x is None ]
    
    if data_format == 'channels_first':
        return tf.concat(concat_inputs, axis=1)
    else: # NHWC
        return tf.concat(concat_inputs, axis=3)



def flow_block(image_pair, image2_2=None, intrinsics=None, prev_predictions=None, data_format='channels_first', kernel_regularizer=None):
    """Creates a flow network
    
    image_pair: Tensor
        Image pair concatenated along the channel axis.

    image2_2: Tensor
        Second image at resolution level 2 (downsampled two times)
        
    intrinsics: Tensor 
        The normalized intrinsic parameters

    prev_predictions: dict of Tensor
        Predictions from the previous depth block
    
    Returns a dict with the predictions
    """
    conv_params = {'data_format':data_format, 'kernel_regularizer':kernel_regularizer}

    # contracting part
    conv1 = convrelu2(name='conv1', inputs=image_pair, num_outputs=(24,32), kernel_size=9, stride=2, **conv_params)

    if prev_predictions is None:
        conv2 = convrelu2(name='conv2', inputs=conv1, num_outputs=(48,64), kernel_size=7, stride=2, **conv_params)
        conv2_1 = convrelu2(name='conv2_1', inputs=conv2, num_outputs=64, kernel_size=3, stride=1, **conv_params)
    else:
        conv2 = convrelu2(name='conv2', inputs=conv1, num_outputs=32, kernel_size=7, stride=2, **conv_params)

        # create warped input
        if data_format=='channels_first':
            prev_depth_nchw = prev_predictions['predict_depth2']
        else:
            prev_depth_nchw = convert_NHWC_to_NCHW(prev_predictions['predict_depth2'])

        _flow_from_depth_motion = sops.depth_to_flow(
            intrinsics = intrinsics,
            depth = prev_depth_nchw,
            rotation = prev_predictions['predict_rotation'],
            translation = prev_predictions['predict_translation'],
            inverse_depth = True,
            normalize_flow = True,
            )
        # set flow vectors to zero if the motion is too large.
        # this also eliminates nan values which can be produced by very bad camera parameters
        flow_from_depth_motion_norm = tf.norm(_flow_from_depth_motion, axis=1, keep_dims=True)
        flow_from_depth_motion_norm = tf.concat((flow_from_depth_motion_norm, flow_from_depth_motion_norm),axis=1)
        tmp_zeros = tf.zeros_like(_flow_from_depth_motion,dtype=tf.float32)
        flow_from_depth_motion =  tf.where( flow_from_depth_motion_norm < 1.0, _flow_from_depth_motion, tmp_zeros)


        image2_2_warped = sops.warp2d(
            input = image2_2 if data_format=='channels_first' else convert_NHWC_to_NCHW(image2_2),
            displacements = flow_from_depth_motion,
            normalized = True,
            border_mode = 'value',
            )
        if data_format=='channels_last':
            flow_from_depth_motion = convert_NCHW_to_NHWC(flow_from_depth_motion)
            image2_2_warped = convert_NCHW_to_NHWC(image2_2_warped)
        extra_inputs = (image2_2_warped, flow_from_depth_motion, prev_predictions['predict_depth2'], prev_predictions['predict_normal2'])

        # stop gradient here
        extra_inputs_concat = tf.stop_gradient(tf.concat(extra_inputs, axis=1 if data_format=='channels_first' else 3))

        conv_extra_inputs = convrelu2(name='conv2_extra_inputs', inputs=extra_inputs_concat, num_outputs=32, kernel_size=3, stride=1, **conv_params)
        conv2_concat = tf.concat((conv2,conv_extra_inputs), axis=1 if data_format=='channels_first' else 3)
        conv2_1 = convrelu2(name='conv2_1', inputs=conv2_concat, num_outputs=64, kernel_size=3, stride=1, **conv_params)
    
    
    conv3 = convrelu2(name='conv3', inputs=conv2_1, num_outputs=(96,128), kernel_size=5, stride=2, **conv_params)
    conv3_1 = convrelu2(name='conv3_1', inputs=conv3, num_outputs=128, kernel_size=3, stride=1, **conv_params)
    
    conv4 = convrelu2(name='conv4', inputs=conv3_1, num_outputs=(192,256), kernel_size=5, stride=2, **conv_params)
    conv4_1 = convrelu2(name='conv4_1', inputs=conv4, num_outputs=256, kernel_size=3, stride=1, **conv_params)
    
    conv5 = convrelu2(name='conv5', inputs=conv4_1, num_outputs=384, kernel_size=5, stride=2, **conv_params)
    conv5_1 = convrelu2(name='conv5_1', inputs=conv5, num_outputs=384, kernel_size=3, stride=1, **conv_params)

    dense_slice_shape = conv5_1.get_shape().as_list()
    if data_format == 'channels_first':
        dense_slice_shape[1] = 96
    else:
        dense_slice_shape[-1] = 96
    units = 1
    for i in range(1,len(dense_slice_shape)):
        units *= dense_slice_shape[i]
    dense5 = tf.layers.dense(
            tf.contrib.layers.flatten(tf.slice(conv5_1, [0,0,0,0], dense_slice_shape)),
            units=units,
            activation=myLeakyRelu,
            kernel_initializer=default_weights_initializer(),
            kernel_regularizer=kernel_regularizer,
            name='dense5'
            )
    print(dense5)
    conv5_1_dense5 = tf.concat((conv5_1,tf.reshape(dense5, dense_slice_shape)),  axis=1 if data_format=='channels_first' else 3)

    
    # expanding part
    with tf.variable_scope('predict_flow5'):
        predict_flowconf5 = _predict_flow(conv5_1_dense5, predict_confidence=True, **conv_params)
    
    with tf.variable_scope('upsample_flow5to4'):
        predict_flowconf5to4 = _upsample_prediction(predict_flowconf5, 2, **conv_params)
   
    with tf.variable_scope('refine4'):
        concat4 = _refine(
            inp=conv5_1_dense5, 
            num_outputs=256, 
            upsampled_prediction=predict_flowconf5to4, 
            features_direct=conv4_1,
            **conv_params,
        )

    with tf.variable_scope('refine3'):
        concat3 = _refine(
            inp=concat4, 
            num_outputs=128, 
            features_direct=conv3_1,
            **conv_params,
        )

    with tf.variable_scope('refine2'):
        concat2 = _refine(
            inp=concat3, 
            num_outputs=64, 
            features_direct=conv2_1,
            **conv_params,
        )

    with tf.variable_scope('predict_flow2'):
        predict_flowconf2 = _predict_flow(concat2, predict_confidence=True, **conv_params)
 
    return { 'predict_flowconf5': predict_flowconf5, 'predict_flowconf2': predict_flowconf2 }


def _predict_depthnormal(inp, predicted_scale=None, predict_normals=True, intermediate_num_outputs=24, data_format='channels_first', **kwargs):
    """Generates the ops for depth and normal prediction
    
    inp: Tensor

    predicted_scale: Tensor
        The predicted scale for scaling the depth values

    predict_normals: bool
        If True the output tensor has 4 channels instead of 1.
        The last three channels are the normals.

    intermediate_num_outputs: Tensor
        Number of filters for the intermediate feature blob

    Returns the depth prediction and the normal predictions separately
    """

    tmp = convrelu(
        inputs=inp,
        num_outputs=intermediate_num_outputs,
        kernel_size=3,
        strides=1,
        name="conv1",
        data_format=data_format,
        **kwargs,
    )
    
    tmp2 = conv2d(
        inputs=tmp,
        num_outputs=4 if predict_normals else 1,
        kernel_size=3,
        strides=1,
        name="conv2",
        data_format=data_format,
        **kwargs,
    )

    if predict_normals:

        predicted_unscaled_depth, predicted_normal = tf.split(value=tmp2, num_or_size_splits=[1,3], axis=1 if data_format=='channels_first' else 3) 

        if not predicted_scale is None:
            batch_size = predicted_scale.get_shape().as_list()[0]
            s = tf.reshape(predicted_scale, [batch_size,1,1,1])
            predicted_depth = s*predicted_unscaled_depth
        else:
            predicted_depth = predicted_unscaled_depth

        return predicted_depth, predicted_normal
    else:
        if not predicted_scale is None:
            predicted_depth = predicted_scale*tmp2
        else:
            predicted_depth = tmp2

        return predicted_depth




def depthmotion_block(image_pair, image2_2, prev_flow2, prev_flowconf2, prev_rotation=None, prev_translation=None, intrinsics=None, data_format='channels_first', kernel_regularizer=None):
    """Creates a depth and motion network
    
    image_pair: Tensor
        Image pair concatenated along the channel axis.
        The tensor format is NCHW with C == 6.

    image2_2: Tensor
        Second image at resolution level 2

    prev_flow2: Tensor
        The output of the flow network. Contains only the flow (2 channels)

    prev_flowconf2: Tensor
        The output of the flow network. Contains flow and flow confidence (4 channels)

    prev_rotation: Tensor
        The previously predicted rotation.
        
    prev_translaion: Tensor
        The previously predicted translation.

    intrinsics: Tensor
        Tensor with the intrinsic camera parameters
        Only required if prev_rotation and prev_translation is not None.
        
    Returns a dictionary with the predictions for depth, normals and motion
    """
    conv_params = {'data_format':data_format, 'kernel_regularizer':kernel_regularizer}
    fc_params = {}
    
    # contracting part
    conv1 = convrelu2(name='conv1', inputs=image_pair, num_outputs=(24,32), kernel_size=9, stride=2, **conv_params)
    
    conv2 = convrelu2(name='conv2', inputs=conv1, num_outputs=32, kernel_size=7, stride=2, **conv_params)
    # create extra inputs
    if data_format=='channels_first':
        image2_2_warped = sops.warp2d(image2_2, prev_flow2, normalized=True, border_mode='value')
    else:
        prev_flow2_nchw = convert_NHWC_to_NCHW(prev_flow2)
        image2_2_warped = convert_NCHW_to_NHWC(sops.warp2d(convert_NHWC_to_NCHW(image2_2), prev_flow2_nchw, normalized=True, border_mode='value'))
        
    extra_inputs = [image2_2_warped, prev_flowconf2]
    if (not prev_rotation is None) and (not prev_translation is None) and (not intrinsics is None):
        if data_format=='channels_first':
            depth_from_flow = sops.flow_to_depth2(
                flow=prev_flow2, 
                intrinsics=intrinsics,
                rotation=prev_rotation,
                translation=prev_translation,
                normalized_flow=True,
                inverse_depth=True,
                )
        else:
            depth_from_flow = convert_NCHW_to_NHWC(sops.flow_to_depth2(
                flow=prev_flow2_nchw, 
                intrinsics=intrinsics,
                rotation=prev_rotation,
                translation=prev_translation,
                normalized_flow=True,
                inverse_depth=True,
                ))
        depth_from_flow = tf.clip_by_value(depth_from_flow, 0.0, 50.0)    

        extra_inputs.append(depth_from_flow)

    concat_extra_inputs = tf.stop_gradient(tf.concat(extra_inputs, axis=1 if data_format=='channels_first' else 3))
    conv_extra_inputs = convrelu2(name='conv2_extra_inputs', inputs=concat_extra_inputs, num_outputs=32, kernel_size=3, stride=1, **conv_params)
    conv2_concat = tf.concat((conv2,conv_extra_inputs),axis=1 if data_format=='channels_first' else 3)
    conv2_1 = convrelu2(name='conv2_1', inputs=conv2_concat, num_outputs=64, kernel_size=3, stride=1, **conv_params)
    
    conv3 = convrelu2(name='conv3', inputs=conv2_1, num_outputs=(96,128), kernel_size=5, stride=2, **conv_params)
    conv3_1 = convrelu2(name='conv3_1', inputs=conv3, num_outputs=128, kernel_size=3, stride=1, **conv_params)
    
    conv4 = convrelu2(name='conv4', inputs=conv3_1, num_outputs=(192,256), kernel_size=5, stride=2, **conv_params)
    conv4_1 = convrelu2(name='conv4_1', inputs=conv4, num_outputs=256, kernel_size=3, stride=1, **conv_params)
    
    conv5 = convrelu2(name='conv5', inputs=conv4_1, num_outputs=384, kernel_size=3, stride=2, **conv_params)
    conv5_1 = convrelu2(name='conv5_1', inputs=conv5, num_outputs=384, kernel_size=3, stride=1, **conv_params)
    
    dense_slice_shape = conv5_1.get_shape().as_list()
    if data_format == 'channels_first':
        dense_slice_shape[1] = 96
    else:
        dense_slice_shape[-1] = 96
    units = 1
    for i in range(1,len(dense_slice_shape)):
        units *= dense_slice_shape[i]
    dense5 = tf.layers.dense(
            tf.contrib.layers.flatten(tf.slice(conv5_1, [0,0,0,0], dense_slice_shape)),
            units=units,
            activation=myLeakyRelu,
            kernel_initializer=default_weights_initializer(),
            kernel_regularizer=kernel_regularizer,
            name='dense5'
            )
    print(dense5)
    conv5_1_dense5 = tf.concat((conv5_1,tf.reshape(dense5, dense_slice_shape)),  axis=1 if data_format=='channels_first' else 3)
    
    # motion prediction part
    motion_conv3 = convrelu2(name='motion_conv3', inputs=conv2_1, num_outputs=64, kernel_size=5, stride=2, **conv_params)
    motion_conv4 = convrelu2(name='motion_conv4', inputs=motion_conv3, num_outputs=64, kernel_size=5, stride=2, **conv_params)
    motion_conv5a = convrelu2(name='motion_conv5a', inputs=motion_conv4, num_outputs=64, kernel_size=3, stride=2, **conv_params)

    motion_conv5b = convrelu(
        name='motion_conv5b',
        inputs=conv5_1_dense5,
        num_outputs=64,
        kernel_size=3,
        strides=1,
        **conv_params,
    )
    motion_conv5_1 = tf.concat((motion_conv5a, motion_conv5b), axis=1 if data_format=='channels_first' else 3)

    if data_format=='channels_last':
        motion_conv5_1 = convert_NHWC_to_NCHW(motion_conv5_1)
    motion_fc1 = tf.layers.dense(
        name='motion_fc1',
        inputs=tf.contrib.layers.flatten(motion_conv5_1),
        units=1024,
        activation=myLeakyRelu,
        kernel_regularizer=kernel_regularizer,
        **fc_params,
    )
    motion_fc2 = tf.layers.dense(
        name='motion_fc2',
        inputs=motion_fc1,
        units=128,
        activation=myLeakyRelu,
        kernel_regularizer=kernel_regularizer,
        **fc_params,
    )
    predict_motion_scale = tf.layers.dense(
        name='motion_fc3',
        inputs=motion_fc2,
        units=7,
        activation=None,
        kernel_regularizer=kernel_regularizer,
        **fc_params,
    )

    predict_rotation, predict_translation, predict_scale = tf.split(value=predict_motion_scale, num_or_size_splits=[3,3,1], axis=1)
    
    # expanding part
    with tf.variable_scope('refine4'):
        concat4 = _refine(
            inp=conv5_1, 
            num_outputs=256, 
            features_direct=conv4_1,
            **conv_params,
        )

    with tf.variable_scope('refine3'):
        concat3 = _refine(
            inp=concat4, 
            num_outputs=128, 
            features_direct=conv3_1,
            **conv_params,
        )

    with tf.variable_scope('refine2'):
        concat2 = _refine(
            inp=concat3, 
            num_outputs=64, 
            features_direct=conv2_1,
            **conv_params,
        )

    with tf.variable_scope('predict_depthnormal2'):
        predict_depth2, predict_normal2 = _predict_depthnormal(concat2, predicted_scale=predict_scale, **conv_params)
 
    return { 
        'predict_depth2': predict_depth2,  
        'predict_normal2': predict_normal2,  
        'predict_rotation': predict_rotation,  
        'predict_translation': predict_translation,  
        'predict_scale': predict_scale,
        }
    



def depth_refine_block(image1, depthmotion_predictions, data_format='channels_first', kernel_regularizer=None ):
    """Creates a refinement network for the depth and normal predictions
    
    image1: Tensor
        The reference image at full resolution.

    depthmotion_predictions: dict of Tensors
        The output of the depthmotion network.

    Returns a dictionary with the predictions
    """
    conv_params = {'data_format': data_format, 'kernel_regularizer':kernel_regularizer}

    # upsample the predicted depth and normals to the same size as the reference image 
    if data_format=='channels_first':
        original_image_size = image1.get_shape().as_list()[-2:]
    else:
        original_image_size = image1.get_shape().as_list()[1:3]
    depth2 = depthmotion_predictions['predict_depth2']
    if data_format=='channels_first':
        depth2_nhwc = convert_NCHW_to_NHWC(depth2) 
    else:
        depth2_nhwc = depth2
    depth2_nhwc_orig_size = tf.image.resize_nearest_neighbor(depth2_nhwc, original_image_size)

    if data_format=='channels_first':
        depth2_orig_size = convert_NHWC_to_NCHW(depth2_nhwc_orig_size)
    else:
        depth2_orig_size = depth2_nhwc_orig_size

    net_inputs = tf.concat((image1, depth2_orig_size), axis=1 if data_format=='channels_first' else 3)

    # contracting part
    conv0 = convrelu(name='conv0', inputs=net_inputs, num_outputs=32, kernel_size=3, strides=1, **conv_params)
    
    conv1 = convrelu(name='conv1', inputs=conv0, num_outputs=64, kernel_size=3, strides=2, **conv_params)
    conv1_1 = convrelu(name='conv1_1', inputs=conv1, num_outputs=64, kernel_size=3, strides=1, **conv_params)
    
    conv2 = convrelu(name='conv2', inputs=conv1_1, num_outputs=128, kernel_size=3, strides=2, **conv_params)
    conv2_1 = convrelu(name='conv2_1', inputs=conv2, num_outputs=128, kernel_size=3, strides=1, **conv_params)

    # expanding part
    with tf.variable_scope('refine1'):
        concat1 = _refine(
            inp=conv2_1, 
            num_outputs=64, 
            features_direct=conv1_1,
            **conv_params,
        )
    
    with tf.variable_scope('refine0'):
        concat0 = _refine(
            inp=concat1, 
            num_outputs=32, 
            features_direct=conv0,
            **conv_params,
        )
    
    with tf.variable_scope('predict_depth0'):
        predict_depth0, predict_normal0 = _predict_depthnormal(concat0, predict_normals=True, intermediate_num_outputs=16, **conv_params)

    return { 'predict_depth0': predict_depth0, 'predict_normal0': predict_normal0 }


