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
from .metrics import compute_motion_errors,evaluate_depth,compute_flow_epe
import h5py
import xarray
import numpy as np
import re
import math
import json
import scipy.misc
import time
import skimage.transform

'''
Functions to evaluate DeMoN results stored as hdf5 files. The results are stored as xarray DataArray converted to json
'''

def write_xarray_json(data, out_file):
    """Writes xarray as json to a file"""
    with open(out_file, 'w') as f:
        json.dump(data.to_dict(), f)
        
def read_xarray_json(in_file):
    """Reads xarray from a json file"""
    with open(in_file, 'r') as f:
        return xarray.DataArray.from_dict(json.load(f))
           
def get_metainfo(data_file):
    """Checks a hdf5 data file for its format and dimensions. 

    data_file: str
        Path to the hdf5 file generated with the test_iterative.py script.

    returns a dictionary with the following keys:
        iterative:  bool, if the file is from an iterative net
        snapshots:  list of str, names od snapshots in the file contain
        iterations: list of net_iterations
        samples:    list of samples
    """
    
    re_iteration = re.compile('.*_(\d+)(\.caffemodel\.h5)?')
    with h5py.File(data_file,'r') as f:
        group_name = list(f.keys())[0]
        iterative_net = bool(re_iteration.match(group_name))
        if iterative_net:
            snapshots = list(f.keys())
            snapshots.sort(key=lambda x: int(re_iteration.match(x).group(1)))
            snapshot_iters = [int(re_iteration.match(x).group(1)) for x in snapshots]
            snapshot_group = f[snapshots[0]]
            samples = list(snapshot_group.keys())
            samples.sort(key=int)
            sample_group = snapshot_group[samples[0]]
            # collect iterations from all prediction datasets
            iterations = set()
            for prediction in ('predicted_depth', 'predicted_normal', 'predicted_motion', 'predicted_flow', 'predicted_conf'):
                if prediction in sample_group:
                    iterations.update( list(sample_group[prediction]) )
            iterations = list(iterations)
            iterations.sort(key=lambda x: (int(x.split('_')[0]),len(x.split('_'))))
        else:
            snapshots = ['snapshot']
            snapshot_iters = [-1]
            iterations = ['0']
            samples = list(f.keys())
            samples.sort(key=int)

    metainfo = {
            'iterative':iterative_net, 
            'snapshots': snapshots, 
            'iterations': iterations, 
            'samples':samples, 
            'snapshot_iters': snapshot_iters, 
            'input_file': data_file,
            }
    return metainfo


def invalidate_points_not_visible_in_second_image(depth, motion, intrinsics):
    """Sets the depth values for the points not visible in the second view to nan

    depth: numpy.ndarray
        array with inverse depth values as stored in the test output h5 files

    motion: numpy.ndarray
        the 6 element motion vector (ANGLEAXIS6)

    intrinsics: numpy.ndarray or None
        the normalized intrinsics vector
        If None we assume intrinsics as in sun3d
    """
    from .helpers import motion_vector_to_Rt, intrinsics_vector_to_K
    from ..dataset_tools.view import View
    from ..dataset_tools.view_tools import compute_visible_points_mask
    #from matplotlib import pyplot as plt
    abs_depth = 1/depth
    R, t = motion_vector_to_Rt(motion.squeeze())

    if intrinsics is None:
        intrinsics = np.array([[0.891, 1.188, 0.5, 0.5]], dtype=np.float32) # sun3d intrinsics
    intrinsics = intrinsics.squeeze()
    K = intrinsics_vector_to_K(intrinsics, depth.shape[-1], depth.shape[-2])
    view1 = View(R=np.eye(3), t=np.zeros((3,)), K=K, image=None, depth=abs_depth, depth_metric='camera_z')
    view2 = View(R=R, t=t, K=K, image=None, depth=abs_depth, depth_metric='camera_z')
    invalid_points = compute_visible_points_mask(view1, view2) == 0
    # tmp = depth.copy()
    depth[invalid_points] = np.nan
    # plt.imshow(np.concatenate((tmp,depth),axis=1))
    # plt.show(block=True)
    
        



def get_data(iterative, results_h5_file, snap, sample, net_iter, gt_h5_file=None, depthmask=False, eigen_crop_gt_and_pred=False):
    """Helper function to read data from the h5 files
    
    iterative: bool
        If true the hdf5 file stores results from multiple iterations.

    results_h5_file: h5py.File
        The file with the network predictions

    snap: str
        Name of the snapshot

    sample: str
        Sample number as string

    net_iter: int
        network iteration

    gt_h5_file: h5py.File
        ground truth h5 file.

    depthmask: bool
        If True the depth values for points not visible in the second image will be masked out
        
    eigen_crop_gt_and_pred: bool
        If true crops images and depth maps to match the evaluation for NYU in Eigen's paper.

    Returns a dictionary with ground truth and predictions for depth, motion and flow.
    """
    data_types = ['motion', 'depth', 'flow', 'normals', 'intrinsics']
    data = {}
    # get ground truth
    if iterative and (gt_h5_file is None):
        sample_group = results_h5_file[snap][sample]
    else:
        if gt_h5_file is None:
            sample_group = results_h5_file[sample]
        else:
            sample_group = gt_h5_file[sample]
            gt_sample_id = sample_group.attrs['sample_id']

    for dt in data_types:
        if dt in sample_group:
            data[dt + '_gt'] = sample_group[dt][:]    
            
    # get predictions
    if iterative:
        sample_group = results_h5_file[snap][sample]
        pr_sample_id = sample_group.attrs['sample_id']
        assert gt_sample_id == pr_sample_id, "sample ids do not match: prediction id='{0}', ground truth id='{1}'".format(pr_sample_id,gt_sample_id)
        for dt in data_types:
            if 'predicted_{0}/{1}'.format(dt,net_iter) in sample_group:
                data[dt + '_pred'] = sample_group['predicted_'+dt][net_iter][:]
    else:
        sample_group = results_h5_file[sample]
        for dt in data_types:
            if ('predicted_'+dt) in sample_group:
                data[dt + '_pred'] = sample_group['predicted_'+dt][:]
        
    for key in data:    
        data[key] = np.squeeze(data[key])
        
    if ('depth_pred' in data) and (data['depth_pred'].shape == (109,147)):
        print('\n >>> Eigen and Fergus detected, cropping the ground truth <<<\n')
        assert(data['depth_gt'].shape == (480,640))
        data['depth_gt'] = data['depth_gt'][23:23+436,27:27+588]
        
    if depthmask and ('motion_gt' in data) and ('depth_gt' in data):
        intrinsics = data['intrinsics'] if 'intrinsics' in data else None
        invalidate_points_not_visible_in_second_image(data['depth_gt'], data['motion_gt'], intrinsics)
    
    # reshape the predictions to GT size if necessary
    if ('depth_gt' in data) and ('depth_pred' in data) and (not (data['depth_gt'].shape == data['depth_pred'].shape)):
        data['depth_pred'] = skimage.transform.resize(data['depth_pred'], data['depth_gt'].shape, order=0, mode='constant', preserve_range=True)
    if ('flow_gt' in data) and ('flow_pred' in data) and (not (data['flow_gt'].shape == data['flow_pred'].shape)):
        data['flow_pred'] = np.transpose(skimage.transform.resize(\
                                np.transpose(data['flow_pred'],(1,2,0)), data['depth_gt'].shape, order=0, mode='constant', preserve_range=True),(2,0,1))
        
    if eigen_crop_gt_and_pred and data['depth_gt'].shape != (436,588):
        assert(data['depth_gt'].shape == (480,640))
        assert(data['depth_pred'].shape == (480,640))
        data['depth_gt'] = data['depth_gt'][23:23+436,27:27+588]
        data['depth_pred'] = data['depth_pred'][23:23+436,27:27+588]
    
    return data
        

def evaluate(results_file, gt_file, depthmask=False, eigen_crop_gt_and_pred=False, depth_scaling='abs'):
    '''
    Compute different error measures given a hdf5 result (prediction) file, and output them as an xarray.
    results_file: str
        Path to the network results (prediction) in hdf5 format.

    gt_file: str
        Path to the hdf5 file with ground truth data stored in the simple test output format

    depthmask: bool
        If True the depth values for points not visible in the second image will be masked out

    eigen_crop_gt_and_pred: bool
        If true crops images and depth maps to match the evaluation for NYU in Eigen's paper.

    depth_scaling: str
        selects a scaling method for the scaled results. E.g. 'abs' scales such that the 
        least squares error for the absolute depth values is minimized.
        
    '''
    depth_pred_max=np.inf

    depth_errors_to_compute = ['l1',
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
    
    errors_to_compute = ['rot_err', 'tran_err', 'tran_angle_err'] + \
                       ['depth_' + e for e in depth_errors_to_compute] + \
                       ['flow_epe', 'camera_baseline']
    
    metainfo = get_metainfo(results_file)
    results = xarray.DataArray(np.zeros((len(metainfo['snapshots']), len(metainfo['iterations']), len(metainfo['samples']), len(errors_to_compute), 2)), 
                             [('snapshot', metainfo['snapshots']),
                              ('iteration', metainfo['iterations']),
                              ('sample', metainfo['samples']),
                              ('errors', errors_to_compute),
                              ('scaled', [False,True])])
    results[:] = np.nan
    
    # save metainfo and evaluation options
    for key,val in metainfo.items():
        results.attrs[key] = val
    results.attrs['gt_file'] = gt_file
    results.attrs['depthmask'] = depthmask
    results.attrs['depth_scaling'] = depth_scaling
    results.attrs['depth_pred_max'] = str(depth_pred_max)

           
    with h5py.File(results_file,'r') as results_f:
        if gt_file:
            gt_f = h5py.File(gt_file,'r')
        else:
            gt_f = None

        t0 = 0
        for nsnap,snap in enumerate(metainfo['snapshots']):
            for nsample,sample in enumerate(metainfo['samples']):            
                for niter,net_iter in enumerate(metainfo['iterations']):
                    if time.time() - t0 > 5:
                        t0 = time.time()
                        print('Processing snapshot %d/%d. sample %d/%d' % \
                                    (nsnap+1, len(metainfo['snapshots']), nsample+1, len(metainfo['samples'])))
                    data = get_data(metainfo['iterative'], results_f, snap, sample, net_iter, gt_h5_file=gt_f, depthmask=depthmask, eigen_crop_gt_and_pred=eigen_crop_gt_and_pred)
                    
                    if ('depth_gt' in data) and ('depth_pred' in data): 
                        #print(data['depth_pred'].dtype, data['depth_pred'][:3,:3], data['depth_gt'].dtype, data['depth_gt'][:3,:3])
                        if 'motion_gt' in data and (not np.any(np.isnan(data['motion_gt']))):
                            translation_gt = data['motion_gt'][-3:]
                            results.loc[snap,net_iter,sample,'camera_baseline'] = np.linalg.norm(translation_gt)
                        else:
                            translation_gt = np.array([1.,0.,0.])                    
                        depth_errs, depth_errs_pred_scaled = evaluate_depth(translation_gt, data['depth_gt'], data['depth_pred'], 
                                                                distances_to_compute=depth_errors_to_compute, inverse_gt=True, inverse_pred=True, 
                                                                depth_scaling=depth_scaling, depth_pred_max=depth_pred_max)
                    
                        for dist in depth_errors_to_compute:
                            results.loc[snap,net_iter,sample,'depth_' + dist,False] = depth_errs[dist]
                            results.loc[snap,net_iter,sample,'depth_' + dist,True] = depth_errs_pred_scaled[dist]
                    
                    if ('motion_gt' in data) and ('motion_pred' in data):
                        normalize_translation = True
                        rot_err, tran_err, tran_angle_err = compute_motion_errors(data['motion_pred'], data['motion_gt'], normalize_translation)
                        results.loc[snap,net_iter,sample,'rot_err'] = rot_err
                        results.loc[snap,net_iter,sample,'tran_err'] = tran_err
                        results.loc[snap,net_iter,sample,'tran_angle_err'] = tran_angle_err
                    
                    if ('flow_gt' in data) and ('flow_pred' in data):
                        flow_epe = compute_flow_epe(data['flow_pred'],data['flow_gt'])
                        results.loc[snap,net_iter,sample,'flow_epe'] = flow_epe
        if gt_file:
            gt_f.close()
             
    return results
    
        


