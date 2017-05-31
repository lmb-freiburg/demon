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
import numpy as np
import math
from minieigen import Quaternion, Vector3

# implements error metrics from Eigen et al. https://arxiv.org/pdf/1406.2283.pdf

def compute_valid_depth_mask(d1, d2=None):
    """Computes the mask of valid values for one or two depth maps
    
    Returns a valid mask that only selects values that are valid depth value 
    in both depth maps (if d2 is given).
    Valid depth values are >0 and finite.
    """
    if d2 is None:
        valid_mask = np.isfinite(d1)
        valid_mask[valid_mask] = (d1[valid_mask] > 0)
    else:
        valid_mask = np.isfinite(d1) & np.isfinite(d2)
        valid_mask[valid_mask] = (d1[valid_mask] > 0) & (d2[valid_mask] > 0)
    return valid_mask


def l1(depth1,depth2):
    """
    Computes the l1 errors between the two depth maps.
    Takes preprocessed depths (no nans, infs and non-positive values)

    depth1:  one depth map
    depth2:  another depth map

    Returns: 
        L1(log)

    """
    assert(np.all(np.isfinite(depth1) & np.isfinite(depth2) & (depth1 > 0) & (depth2 > 0)))
    diff = depth1 - depth2
    num_pixels = float(diff.size)
    
    if num_pixels == 0:
        return np.nan
    else:
        return np.sum(np.absolute(diff)) / num_pixels
    

def l1_inverse(depth1,depth2):
    """
    Computes the l1 errors between inverses of two depth maps.
    Takes preprocessed depths (no nans, infs and non-positive values)

    depth1:  one depth map
    depth2:  another depth map

    Returns: 
        L1(log)

    """
    assert(np.all(np.isfinite(depth1) & np.isfinite(depth2) & (depth1 > 0) & (depth2 > 0)))
    diff = np.reciprocal(depth1) - np.reciprocal(depth2)
    num_pixels = float(diff.size)
    
    if num_pixels == 0:
        return np.nan
    else:
        return np.sum(np.absolute(diff)) / num_pixels


def rmse_log(depth1,depth2):
    """
    Computes the root min square errors between the logs of two depth maps.
    Takes preprocessed depths (no nans, infs and non-positive values)

    depth1:  one depth map
    depth2:  another depth map

    Returns: 
        RMSE(log)

    """
    assert(np.all(np.isfinite(depth1) & np.isfinite(depth2) & (depth1 > 0) & (depth2 > 0)))
    log_diff = np.log(depth1) - np.log(depth2)
    num_pixels = float(log_diff.size)
    
    if num_pixels == 0:
        return np.nan
    else:
        return np.sqrt(np.sum(np.square(log_diff)) / num_pixels)
    

def rmse(depth1,depth2):
    """
    Computes the root min square errors between the two depth maps.
    Takes preprocessed depths (no nans, infs and non-positive values)

    depth1:  one depth map
    depth2:  another depth map

    Returns: 
        RMSE(log)

    """
    assert(np.all(np.isfinite(depth1) & np.isfinite(depth2) & (depth1 > 0) & (depth2 > 0)))
    diff = depth1 - depth2
    num_pixels = float(diff.size)
    
    if num_pixels == 0:
        return np.nan
    else:
        return np.sqrt(np.sum(np.square(diff)) / num_pixels)
    

def scale_invariant(depth1,depth2):
    """
    Computes the scale invariant loss based on differences of logs of depth maps.
    Takes preprocessed depths (no nans, infs and non-positive values)

    depth1:  one depth map
    depth2:  another depth map

    Returns: 
        scale_invariant_distance

    """
    # sqrt(Eq. 3)
    assert(np.all(np.isfinite(depth1) & np.isfinite(depth2) & (depth1 > 0) & (depth2 > 0)))
    log_diff = np.log(depth1) - np.log(depth2)
    num_pixels = float(log_diff.size)
    
    if num_pixels == 0:
        return np.nan
    else:
        return np.sqrt(np.sum(np.square(log_diff)) / num_pixels - np.square(np.sum(log_diff)) / np.square(num_pixels))


def abs_relative(depth_pred,depth_gt):
    """
    Computes relative absolute distance.
    Takes preprocessed depths (no nans, infs and non-positive values)

    depth_pred:  depth map prediction
    depth_gt:    depth map ground truth

    Returns: 
        abs_relative_distance

    """
    assert(np.all(np.isfinite(depth_pred) & np.isfinite(depth_gt) & (depth_pred > 0) & (depth_gt > 0)))
    diff = depth_pred - depth_gt
    num_pixels = float(diff.size)

    if num_pixels == 0:
        return np.nan
    else:
        return np.sum(np.absolute(diff) / depth_gt) / num_pixels


def avg_log10(depth1,depth2):
    """
    Computes average log_10 error (Liu, Neural Fields, 2015).
    Takes preprocessed depths (no nans, infs and non-positive values)

    depth1:  one depth map
    depth2:  another depth map

    Returns:
        abs_relative_distance

    """
    assert(np.all(np.isfinite(depth1) & np.isfinite(depth2) & (depth1 > 0) & (depth2 > 0)))
    log_diff = np.log10(depth1) - np.log10(depth2)
    num_pixels = float(log_diff.size)

    if num_pixels == 0:
        return np.nan
    else:
        return np.sum(np.absolute(log_diff)) / num_pixels
    

def sq_relative(depth_pred,depth_gt):
    """
    Computes relative squared distance.
    Takes preprocessed depths (no nans, infs and non-positive values)

    depth_pred:  depth map prediction
    depth_gt:    depth map ground truth

    Returns: 
        squared_relative_distance

    """
    assert(np.all(np.isfinite(depth_pred) & np.isfinite(depth_gt) & (depth_pred > 0) & (depth_gt > 0)))
    diff = depth_pred - depth_gt
    num_pixels = float(diff.size)
    
    if num_pixels == 0:
        return np.nan
    else:
        return np.sum(np.square(diff) / depth_gt) / num_pixels
    

def ratio_threshold(depth1, depth2, threshold):
    """
    Computes the percentage of pixels for which the ratio of the two depth maps is less than a given threshold.
    Takes preprocessed depths (no nans, infs and non-positive values)

    depth1:  one depth map
    depth2:  another depth map

    Returns: 
        percentage of pixels with ratio less than the threshold

    """
    assert(threshold > 0.)
    assert(np.all(np.isfinite(depth1) & np.isfinite(depth2) & (depth1 > 0) & (depth2 > 0)))
    log_diff = np.log(depth1) - np.log(depth2)
    num_pixels = float(log_diff.size)
    
    if num_pixels == 0:
        return np.nan
    else:
        return float(np.sum(np.absolute(log_diff) < np.log(threshold))) / num_pixels
    

def compute_errors(depth_pred, depth_gt, distances_to_compute=None):
    """
    Computes different distance measures between two depth maps.

    depth_pred:           depth map prediction
    depth_gt:             depth map ground truth
    distances_to_compute: which distances to compute

    Returns: 
        a dictionary with computed distances, and the number of valid pixels

    """

    valid_mask = compute_valid_depth_mask(depth_pred, depth_gt)
    depth_pred = depth_pred[valid_mask]
    depth_gt   = depth_gt[valid_mask]
    num_valid  = np.sum(valid_mask)
    
    if distances_to_compute is None:
        distances_to_compute = ['l1',
                                'l1_inverse',
                                'scale_invariant',
                                'abs_relative',
                                'sq_relative',
                                'avg_log10',
                                'rmse_log',
                                'rmse',
                                'ratio_threshold_1.25',
                                'ratio_threshold_1.5625',
                                'ratio_threshold_1.953125']
    
    results = {'num_valid': num_valid}
    for dist in distances_to_compute:
        if dist.startswith('ratio_threshold'):
            threshold = float(dist.split('_')[-1])
            results[dist] = ratio_threshold(depth_pred,depth_gt,threshold)
        else:
            results[dist] = globals()[dist](depth_pred,depth_gt)
        
    return results


def compute_depth_scale_factor(depth1, depth2, depth_scaling='abs'):
    """
    Computes the scale factor for depth1 to minimize the least squares error to depth2
    """
    
    assert(np.all(np.isfinite(depth1) & np.isfinite(depth2) & (depth1 > 0) & (depth2 > 0)))
    
    
    if depth_scaling == 'abs':
        # minimize MSE on depth
        d1d1 = np.multiply(depth1,depth1)
        d1d2 = np.multiply(depth1,depth2)
        mask = compute_valid_depth_mask(d1d2)
        sum_d1d1 = np.sum(d1d1[mask])
        sum_d1d2 = np.sum(d1d2[mask])
        if sum_d1d1 > 0.:
            scale = sum_d1d2/sum_d1d1
        else:
            print('compute_depth_scale_factor: Norm=0 during scaling')
            scale = 1.
    elif depth_scaling == 'log':    
        # minimize MSE on log depth
        log_diff = np.log(depth2) - np.log(depth1)
        scale = np.exp(np.mean(log_diff))
    elif depth_scaling == 'inv':    
        # minimize MSE on inverse depth
        d1d1 = np.multiply(np.reciprocal(depth1),np.reciprocal(depth1))
        d1d2 = np.multiply(np.reciprocal(depth1),np.reciprocal(depth2))
        mask = compute_valid_depth_mask(d1d2)
        sum_d1d1 = np.sum(d1d1[mask])
        sum_d1d2 = np.sum(d1d2[mask])
        if sum_d1d1 > 0.:
            scale = np.reciprocal(sum_d1d2/sum_d1d1)
        else:
            print('compute_depth_scale_factor: Norm=0 during scaling')
            scale = 1.
    else:
        raise Exception('Unknown depth scaling method')
        
    return scale


def evaluate_depth( translation_gt, depth_gt_in, depth_pred_in, 
    distances_to_compute=None, inverse_gt=True, inverse_pred=True, 
    depth_scaling='abs', depth_pred_max=np.inf ):
    """
    Computes different error measures for the inverse depth map without scaling and with scaling.

    translation_gt: 1d numpy array with [tx,ty,tz]
        The translation that corresponds to the ground truth depth

    depth_gt: 2d numpy array
        This is the ground truth depth
        
    depth_pred: depth prediction being evaluated
    
    distances_to_compute: which distances to compute

    returns (err, err_after_scaling)
        errs is the dictionary of errors without optimally scaling the prediction

        errs_pred_scaled is the dictionary of errors after minimizing 
        the least squares error by scaling the prediction
    """
    
    valid_mask = compute_valid_depth_mask(depth_pred_in, depth_gt_in)
    depth_pred = depth_pred_in[valid_mask]
    depth_gt   = depth_gt_in[valid_mask]
    if inverse_gt:
        depth_gt   = np.reciprocal(depth_gt)
    if inverse_pred:
        depth_pred = np.reciprocal(depth_pred)
    
    #if depth_pred_max < np.inf:
        #depth_pred[depth_pred>depth_pred_max] = depth_pred_max
    
    # we need to scale the ground truth depth if the translation is not normalized
    translation_norm = np.sqrt(translation_gt.dot(translation_gt))
    scale_gt_depth = not np.isclose(1.0, translation_norm)
    if scale_gt_depth:
        depth_gt_scaled = depth_gt / translation_norm
    else:
        depth_gt_scaled = depth_gt
            
    errs = compute_errors(depth_pred,depth_gt_scaled,distances_to_compute)
            
    # minimize the least squares error and compute the errors again
    scale = compute_depth_scale_factor(depth_pred,depth_gt_scaled, depth_scaling=depth_scaling)
    depth_pred_scaled = depth_pred*scale
            
    errs_pred_scaled = compute_errors(depth_pred_scaled,depth_gt_scaled,distances_to_compute)
            
    return errs, errs_pred_scaled


def compute_flow_epe(flow1, flow2):
    """Computes the average endpoint error between the two flow fields"""
    diff = flow1 - flow2
    epe = np.sqrt(diff[0,:,:]**2 + diff[1,:,:]**2)
    # mask out invalid epe values
    valid_mask = compute_valid_depth_mask(epe) 
    epe = epe[valid_mask]
    if epe.size > 0:
        return np.mean(epe)
    else:
        return np.nan


def compute_motion_errors(predicted_motion, gt_motion, normalize_translations):
    """
    Computes the error of the motion.

    predicted_motion: 1d numpy array with 6 elements
        the motions as [aa1, aa2, aa3, tx, ty, tz]
        aa1,aa2,aa3 is an angle axis representation.
        The angle is the norm of the axis

    gt_motion: 1d numpy array with 6 elements
        ground truth motion in the same format as the predicted motion

    normalize_translations: bool
        If True then translations will be normalized before computing the error

    Returns
     rotation angular distance in radian
     tranlation distance of the normalized translations
     tranlation angular distance in radian
    """
    def _numpy_to_Vector3(arr):
        tmp = arr.astype(np.float64)
        return Vector3(tmp[0],tmp[1],tmp[2])

    gt_axis = _numpy_to_Vector3(gt_motion[0:3])
    gt_angle = gt_axis.norm()
    if gt_angle < 1e-6:
        gt_angle = 0
        gt_axis = Vector3(1,0,0)
    else:
        gt_axis.normalize()
    gt_q = Quaternion(gt_angle,gt_axis)

    predicted_axis = _numpy_to_Vector3(predicted_motion[0:3])
    predicted_angle = predicted_axis.norm()
    if predicted_angle < 1e-6:
        predicted_angle = 0
        predicted_axis = Vector3(1,0,0)
    else:
        predicted_axis.normalize()
    predicted_axis.normalize()
    predicted_q =  Quaternion(predicted_angle,predicted_axis)

    rotation_angle_dist = gt_q.angularDistance(predicted_q)
    
    gt_trans = _numpy_to_Vector3(gt_motion[3:6])
    if normalize_translations:
        gt_trans.normalize()
    predicted_trans = _numpy_to_Vector3(predicted_motion[3:6])
    if normalize_translations and predicted_trans.norm() > 1e-6:
        predicted_trans.normalize()
    translation_dist = (gt_trans-predicted_trans).norm()
    
    translation_angle_diff = math.acos(np.clip(gt_trans.dot(predicted_trans),-1,1))
    
    return np.rad2deg(rotation_angle_dist), translation_dist, np.rad2deg(translation_angle_diff)


