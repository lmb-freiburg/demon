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
from .helpers import *


def l1_loss(x, epsilon):
    """L1 loss

    Returns a scalar tensor with the loss
    """
    with tf.name_scope("l1_loss"):
        return tf.reduce_sum(tf.sqrt(x**2 + epsilon))


def pointwise_l2_loss(inp, gt, epsilon, data_format='NCHW'):
    """Computes the pointwise unsquared l2 loss.
    The input tensors must use the format NCHW. 
    This loss ignores nan values. 
    The loss is normalized by the number of pixels.
    
    inp: Tensor
        This is the prediction.
        
    gt: Tensor
        The ground truth with the same shape as 'inp'
        
    epsilon: float
        The epsilon value to avoid division by zero in the gradient computation
    """
    with tf.name_scope('pointwise_l2_loss'):
        gt_ = tf.stop_gradient(gt)
        diff = sops.replace_nonfinite(inp-gt_)
        if data_format == 'NCHW':
            return tf.reduce_mean(tf.sqrt(tf.reduce_sum(diff**2, axis=1)+epsilon))
        else: # NHWC
            return tf.reduce_mean(tf.sqrt(tf.reduce_sum(diff**2, axis=3)+epsilon))



def scale_invariant_gradient( inp, deltas, weights, epsilon=0.001):
    """Computes the scale invariant gradient images
    
    inp: Tensor
        
    deltas: list of int
      The pixel delta for the difference. 
      This vector must be the same length as weight.

    weights: list of float
      The weight factor for each difference.
      This vector must be the same length as delta.

    epsilon: float
      epsilon value for avoiding division by zero
        
    """
    assert len(deltas)==len(weights)

    sig_images = []
    for delta, weight in zip(deltas,weights):
        sig_images.append(sops.scale_invariant_gradient(inp, deltas=[delta], weights=[weight], epsilon=epsilon))
    return tf.concat(sig_images,axis=1)



def scale_invariant_gradient_loss( inp, gt, epsilon ):
    """Computes the scale invariant gradient loss

    inp: Tensor
        Tensor with the scale invariant gradient images computed on the prediction

    gt: Tensor
        Tensor with the scale invariant gradient images computed on the ground truth

    epsilon: float
      epsilon value for avoiding division by zero
    """
    num_channels_inp = inp.get_shape().as_list()[1]
    num_channels_gt = gt.get_shape().as_list()[1]
    assert num_channels_inp%2==0
    assert num_channels_inp==num_channels_gt

    tmp = []
    for i in range(num_channels_inp//2):
        tmp.append(pointwise_l2_loss(inp[:,i*2:i*2+2,:,:], gt[:,i*2:i*2+2,:,:], epsilon))

    return tf.add_n(tmp)




def flow_loss_block(
    gt_flow2, 
    gt_flow5, 
    gt_flow2_sig, 
    pr_flow2, 
    pr_flow5, 
    pr_conf2, 
    pr_conf5, 
    flow_weight, 
    conf_weight, 
    flow_sig_weight, 
    conf_sig_weight, 
    conf_diff_scale=1,
    level5_factor=0.5,
    loss_prefix='',
    ):
    """Adds loss operations to the flow outputs

    gt_flow2: ground truth flow at resolution level 2
    gt_flow5: ground truth flow at resolution level 5
    gt_flow2_sig: the scale invariant gradient of the ground truth flow at resolution level 2
    pr_flow2: predicted flow at resolution level 2
    pr_flow5: predicted flow at resolution level 5
    pr_conf2: predicted confidence at resolution level 2
    pr_conf5: predicted confidence at resolution level 5
    flow_weight: the weight for the 'absolute' loss on the flows
    conf_weight: the weight for the 'absolute' loss on the flow confidence
    flow_sig_weight: the weight for the loss on the scale invariant gradient images of the flow
    conf_sig_weight: the weight for the loss on the scale invariant gradient images of the confidence
    conf_diff_scale: scale factor for the absolute differences in the conf map computation
    level5_factor: factor for the losses at the smaller resolution level 5. affects losses on pr_flow5 and pr_conf5.
    loss_prefix: prefix name for the loss in the returned dict e.g. 'netFlow1_'

    Returns a dictionary with the losses
    """
    losses = {}
    epsilon = 0.00001

    loss_flow5 = (level5_factor*flow_weight) * pointwise_l2_loss(pr_flow5, gt_flow5, epsilon=epsilon)
    losses['loss_flow5'] = loss_flow5
    loss_flow2 = (flow_weight) * pointwise_l2_loss(pr_flow2, gt_flow2, epsilon=epsilon)
    losses['loss_flow2'] = loss_flow2

    loss_flow5_unscaled = pointwise_l2_loss(pr_flow5, gt_flow5, epsilon=0)
    losses['loss_flow5_unscaled'] = loss_flow5_unscaled
    loss_flow2_unscaled = pointwise_l2_loss(pr_flow2, gt_flow2, epsilon=0)
    losses['loss_flow2_unscaled'] = loss_flow2_unscaled

    # ground truth confidence maps
    conf2 = compute_confidence_map(pr_flow2, gt_flow2, conf_diff_scale)
    conf5 = compute_confidence_map(pr_flow5, gt_flow5, conf_diff_scale)

    if not pr_conf5 is None: 
        loss_conf5 = (level5_factor*conf_weight) * pointwise_l2_loss(pr_conf5, conf5, epsilon=epsilon)
        losses['loss_conf5'] = loss_conf5  
        loss_conf5_unscaled = pointwise_l2_loss(pr_conf5, conf5, epsilon=0)
        losses['loss_conf5_unscaled'] = loss_conf5_unscaled  
    if not pr_conf2 is None:
        loss_conf2 = conf_weight * pointwise_l2_loss(pr_conf2, conf2, epsilon=epsilon)
        losses['loss_conf2'] = loss_conf2  
        loss_conf2_unscaled = pointwise_l2_loss(pr_conf2, conf2, epsilon=0)
        losses['loss_conf2_unscaled'] = loss_conf2_unscaled  


    sig_params = {'deltas':[1,2,4,8,16], 'weights':[1,1,1,1,1], 'epsilon': 0.001}

    if not flow_sig_weight is None:
        pr_flow2_sig = scale_invariant_gradient(pr_flow2, **sig_params)
        loss_flow2_sig = flow_sig_weight * pointwise_l2_loss(pr_flow2_sig, gt_flow2_sig, epsilon=epsilon)
        losses['loss_flow2_sig'] = loss_flow2_sig  
        loss_flow2_sig_unscaled = pointwise_l2_loss(pr_flow2_sig, gt_flow2_sig, epsilon=0)
        losses['loss_flow2_sig_unscaled'] = loss_flow2_sig_unscaled  

    if not conf_sig_weight is None and not pr_conf2 is None:
        pr_conf2_sig = scale_invariant_gradient(pr_conf2, **sig_params)
        conf2_sig = scale_invariant_gradient(conf2, **sig_params)
        loss_conf2_sig = conf_sig_weight * pointwise_l2_loss(pr_conf2_sig, conf2_sig, epsilon=epsilon)
        losses['loss_conf2_sig'] = loss_conf2_sig  
        loss_conf2_sig_unscaled = pointwise_l2_loss(pr_conf2_sig, conf2_sig, epsilon=0)
        losses['loss_conf2_sig_unscaled'] = loss_conf2_sig_unscaled  

    # add prefix and return
    return { loss_prefix+k: losses[k] for k in losses }
        




def depthnormal_loss_block(
    gt_depth2, 
    gt_depth2_sig, 
    gt_normal2,
    gt_rotation,
    gt_translation,
    pr_depth2,
    pr_normal2,
    pr_rotation,
    pr_translation,
    depth_weight, 
    depth_sig_weight, 
    normal_weight, 
    rotation_weight,
    translation_weight,
    translation_factor,
    loss_prefix='' ):
    """Adds loss operations to the flow outputs

    gt_depth2: ground truth depth at resolution level 2
    gt_depth2_sig: the scale invariant gradient of the ground truth depth at resolution level 2
    gt_normal2: ground truth normals at resolution level 2
    pr_depth2: predicted depth at resolution level 2
    pr_normal2: predicted normals at resolution level 2
    depth_weight: the weight for the 'absolute' loss on the depth
    depth_sig_weight: the weight for the loss on the scale invariant gradient image of the depth
    normal_weight: the weight for the loss on the normals
    rotation_weight: the weight for the loss on the rotation
    translation_weight: the weight for the loss on the translation
    translation_factor: additional factor on the translation loss
    loss_prefix: prefix name for the loss in the summary e.g. 'netDM1'

    Returns a dictionary with the losses
    """
    losses = {}
    batch_size = pr_depth2.get_shape().as_list()[0]
    epsilon = 0.00001
    sig_params = {'deltas':[1,2,4,8,16], 'weights':[1,1,1,1,1], 'epsilon':0.01}

    loss_depth2 = depth_weight* pointwise_l2_loss(pr_depth2, gt_depth2, epsilon=epsilon)

    pr_depth2_sig = scale_invariant_gradient(pr_depth2, **sig_params)
    loss_depth2_sig = depth_sig_weight* pointwise_l2_loss(pr_depth2_sig, gt_depth2_sig, epsilon=epsilon)
    loss_depth2_sig_unscaled = pointwise_l2_loss(pr_depth2_sig, gt_depth2_sig, epsilon=0)


    loss_normal2 = normal_weight* pointwise_l2_loss(pr_normal2, gt_normal2, epsilon=epsilon)

    losses['loss_depth2'] = loss_depth2
    losses['loss_depth2_sig'] = loss_depth2_sig
    losses['loss_depth2_sig_unscaled'] = loss_depth2_sig_unscaled
    losses['loss_normal2'] = loss_normal2

    # motion losses
    loss_rotation = (rotation_weight/batch_size)*l1_loss(pr_rotation-gt_rotation, epsilon=epsilon)
    loss_translation_no_factor = (translation_weight/batch_size)*l1_loss(pr_translation-gt_translation, epsilon=epsilon)
    loss_translation = translation_factor*loss_translation_no_factor
    rot_transl_loss_ratio = loss_rotation/loss_translation_no_factor

    losses['loss_rotation'] = loss_rotation
    losses['loss_translation'] = loss_translation
    losses['loss_translation_no_factor'] = loss_translation_no_factor
    losses['rot_transl_loss_ratio'] = rot_transl_loss_ratio
    
    # add prefix and return
    return { loss_prefix+k: losses[k] for k in losses }


def depth_refine_loss_block(
    gt_depth0, 
    gt_depth0_sig, 
    gt_normal0,
    pr_depth0,
    pr_normal0,
    depth_weight, 
    depth_sig_weight, 
    normal_weight, 
    loss_prefix='' ):
    """Adds loss operations to the flow outputs

    gt_depth0: ground truth depth at resolution level 0
    gt_depth0_sig: the scale invariant gradient of the ground truth depth at resolution level 0
    gt_normal0: ground truth normals at resolution level 0
    pr_depth0: predicted depth at resolution level 0
    pr_normal0: predicted normals at resolution level 0
    depth_weight: the weight for the 'absolute' loss on the depth
    depth_sig_weight: the weight for the loss on the scale invariant gradient image of the depth
    normal_weight: the weight for the loss on the normals
    loss_prefix: prefix name for the loss in the summary e.g. 'netRefine'

    Returns a dictionary with the losses
    """
    losses = {}
    epsilon = 0.00001
    sig_params = {'deltas':[1,2,4,8,16], 'weights':[1,1,1,1,1], 'epsilon':0.01}

    loss_depth0 = depth_weight* pointwise_l2_loss(pr_depth0, gt_depth0, epsilon=epsilon)

    pr_depth0_sig = scale_invariant_gradient(pr_depth0, **sig_params)
    loss_depth0_sig = depth_sig_weight* pointwise_l2_loss(pr_depth0_sig, gt_depth0_sig, epsilon=epsilon)
    loss_depth0_sig_unscaled = pointwise_l2_loss(pr_depth0_sig, gt_depth0_sig, epsilon=0)


    loss_normal0 = normal_weight* pointwise_l2_loss(pr_normal0, gt_normal0, epsilon=epsilon)

    losses['loss_depth0'] = loss_depth0
    losses['loss_depth0_sig'] = loss_depth0_sig
    losses['loss_depth0_sig_unscaled'] = loss_depth0_sig_unscaled
    losses['loss_normal0'] = loss_normal0

    # add prefix and return
    return { loss_prefix+k: losses[k] for k in losses }



def prepare_ground_truth_tensors(depth, rotation, translation, intrinsics):
    """Computes ground truth tensors at lower resolution and scale invariant gradient (sig)
    images of some ground truth tensors.
    
    depth: Tensor
        depth map with inverse depth values
        
    rotation: Tensor
        rotation in angle axis format with 3 elements

    translation: Tensor
        the camera translation

    intrinsics: Tensor
        Tensor with the intrinsic camera parameters

    Returns a dictionary with ground truth data for depth, normal and flow for
    different resolutions.
    """
    depth1, depth2, depth3, depth4, depth5 = recursive_median_downsample(depth,5)
    flow0 = sops.depth_to_flow(depth, intrinsics, rotation, translation, inverse_depth=True, normalize_flow=True, name='DepthToFlow0')
    flow2 = sops.depth_to_flow(depth2, intrinsics, rotation, translation, inverse_depth=True, normalize_flow=True, name='DepthToFlow2')
    flow5 = sops.depth_to_flow(depth5, intrinsics, rotation, translation, inverse_depth=True, normalize_flow=True, name='DepthToFlow5')
    
    normal0 = sops.depth_to_normals(depth, intrinsics, inverse_depth=True)
    normal2 = sops.depth_to_normals(depth2, intrinsics, inverse_depth=True)
    
    sig_params = {'deltas':[1,2,4,8,16], 'weights':[1,1,1,1,1], 'epsilon': 0.001}

    depth0_sig = scale_invariant_gradient(depth, **sig_params)
    depth2_sig = scale_invariant_gradient(depth2, **sig_params)
    flow2_sig = scale_invariant_gradient(flow2, **sig_params)
    
    return {
            'depth0': depth, 
            'depth0_sig': depth0_sig, 
            'depth2': depth2, 
            'depth2_sig': depth2_sig, 
            'flow0': flow0, 
            'flow2': flow2, 
            'flow2_sig': flow2_sig, 
            'flow5': flow5, 
            'normal0': normal0,
            'normal2': normal2,
            }



def compute_confidence_map(predicted_flow, gt_flow, scale=1):
    """Computes the ground truth confidence map as c_gt = exp(-s|f_pr-f_gt|) 
    
    predict_flow: Tensor
        The predicted flow
        
    gt_flow: Tensor
        The ground truth flow

    scale: float
        Scale factor for the absolute differences
    """
    with tf.name_scope('compute_confidence_map'):
        return tf.exp(-scale*tf.abs(predicted_flow - gt_flow))
   

