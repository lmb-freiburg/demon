#
# This script computes the depth and motion errors for the network predictions.
#
# Note that numbers are not identical to the values reported in the paper, due 
# to implementation differences between the caffe and tensorflow version.
#
# Running this script requires about 4gb of disk space.
#
# This script expects the test datasets in the folder ../datasets
# Use the provided script in ../datasets for downloading the data.
#
import os
import sys
import json
import h5py
import xarray
import numpy as np
import lmbspecialops as sops
import tensorflow as tf

examples_dir = os.path.dirname(__file__)
weights_dir = os.path.join(examples_dir,'..','weights')
sys.path.insert(0, os.path.join(examples_dir, '..', 'python'))

from depthmotionnet.datareader import *
from depthmotionnet.networks_original import *
from depthmotionnet.helpers import convert_NCHW_to_NHWC, convert_NHWC_to_NCHW
from depthmotionnet.evaluation import *


def create_ground_truth_file(dataset, dataset_dir):
    """Creates a hdf5 file with the ground truth test data
    
    dataset: str
        name of the dataset
    dataset_dir: str
        path to the directory containing the datasets

    Returns the path to the created file
    """
    ds = dataset
    # destination file
    ground_truth_file = '{0}_ground_truth.h5'.format(ds)
    
    if os.path.isfile(ground_truth_file):
        return ground_truth_file # skip existing files
    
    print('creating {0}'.format(ground_truth_file))
    
    # data types requested from the reader op
    data_tensors_keys = ('IMAGE_PAIR', 'MOTION', 'DEPTH', 'INTRINSICS')

    reader_params = {
         'batch_size': 1,
         'test_phase': True,   # deactivates randomization
         'builder_threads': 1, # must be 1 in test phase
         'inverse_depth': True,
         'motion_format': 'ANGLEAXIS6',
         # True is also possible here. If set to True we store ground truth with 
         # precomputed normalization. False keeps the original information.
         'norm_trans_scale_depth': False,
         # original data resolution
         'scaled_height': 480,
         'scaled_width': 640,
         'scene_pool_size': 5, 
         # no augmentation
         'augment_rot180': 0,
         'augment_mirror_x': 0,
         'top_output': data_tensors_keys,
         'source': [{'path': os.path.join(dataset_dir,'{0}_test.h5'.format(ds))}],
        }

    reader_tensors = multi_vi_h5_data_reader(len(data_tensors_keys), json.dumps(reader_params))
    
    # create a dict to make the distinct data tensors accessible via keys
    data_dict = dict(zip(data_tensors_keys,reader_tensors[2]))
    info_tensor = reader_tensors[0]
    sample_ids_tensor = reader_tensors[1]
    rotation_tensor, translation_tensor = tf.split(data_dict['MOTION'], 2, axis=1)

    flow_tensor = sops.depth_to_flow(data_dict['DEPTH'], data_dict['INTRINSICS'], rotation_tensor, translation_tensor, inverse_depth=True, normalize_flow=True)

    gpu_options = tf.GPUOptions()
    gpu_options.per_process_gpu_memory_fraction=0.8 # leave some memory to other processes
    session = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options))


    fetch_dict = {'INFO': info_tensor, 'SAMPLE_IDS': sample_ids_tensor, 'FLOW': flow_tensor}
    fetch_dict.update(data_dict)

    with h5py.File(ground_truth_file) as f:

        number_of_test_iterations = 1 # will be set to the correct value in the while loop
        iteration = 0
        while iteration < number_of_test_iterations:
            
            data =  session.run(fetch_dict)
            
            # get number of iterations from the info vector
            number_of_test_iterations = int(data['INFO'][0])

            # write ground truth data to the file
            group = f.require_group(str(iteration))
            group['image_pair'] = data['IMAGE_PAIR'][0]
            group['depth'] = data['DEPTH'][0]
            group['motion'] = data['MOTION'][0]
            group['flow'] = data['FLOW'][0]
            group['intrinsics'] = data['INTRINSICS'][0]
            
            # save sample id as attribute of the group.
            # the evaluation code will use this to check if prediction and ground truth match.
            sample_id = (''.join(map(chr, data['SAMPLE_IDS']))).strip()
            group.attrs['sample_id'] = np.string_(sample_id)
            iteration += 1
            
    del session
    tf.reset_default_graph()
    return ground_truth_file



def create_prediction_file(dataset, dataset_dir):
    """Creates a hdf5 file with the predictions
    
    dataset: str
        name of the dataset
    dataset_dir: str
        path to the directory containing the datasets

    Returns the path to the created file
    """
  
    if tf.test.is_gpu_available(True):
        data_format='channels_first'
    else: # running on cpu requires channels_last data format
        data_format='channels_last'
    print('Using data_format "{0}"'.format(data_format))

    ds = dataset
    # destination file
    prediction_file = '{0}_prediction.h5'.format(ds)

    # data types requested from the reader op
    data_tensors_keys = ('IMAGE_PAIR', 'MOTION', 'DEPTH', 'INTRINSICS')

    reader_params = {
             'batch_size': 1,
             'test_phase': True,   # deactivates randomization
             'builder_threads': 1, # must be 1 in test phase
             'inverse_depth': True,
             'motion_format': 'ANGLEAXIS6',
             'norm_trans_scale_depth': True,
             # inpu resolution for demon
             'scaled_height': 192,
             'scaled_width': 256,
             'scene_pool_size': 5, 
             # no augmentation
             'augment_rot180': 0,
             'augment_mirror_x': 0,
             'top_output': data_tensors_keys,
             'source': [{'path': os.path.join(dataset_dir,'{0}_test.h5'.format(ds))}],
            }

    reader_tensors = multi_vi_h5_data_reader(len(data_tensors_keys), json.dumps(reader_params))
    
    # create a dict to make the distinct data tensors accessible via keys
    data_dict = dict(zip(data_tensors_keys,reader_tensors[2]))
    info_tensor = reader_tensors[0]
    sample_ids_tensor = reader_tensors[1]
    image1, image2 = tf.split(data_dict['IMAGE_PAIR'],2,axis=1)
    
    # downsample second image
    image2_2 = sops.median3x3_downsample(sops.median3x3_downsample(image2))

    gpu_options = tf.GPUOptions()
    gpu_options.per_process_gpu_memory_fraction=0.8 # leave some memory to other processes
    session = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options))

    
    # init networks
    bootstrap_net = BootstrapNet(session, data_format)
    iterative_net = IterativeNet(session, data_format)
    refine_net = RefinementNet(session, data_format)

    session.run(tf.global_variables_initializer())

    # load weights
    saver = tf.train.Saver()
    saver.restore(session,os.path.join(weights_dir,'demon_original'))

    fetch_dict = {
        'INFO': info_tensor,
        'SAMPLE_IDS': sample_ids_tensor,
        'image1': image1,
        'image2_2': image2_2,
    }
    fetch_dict.update(data_dict)

    if data_format == 'channels_last':
        for k in ('image1', 'image2_2', 'IMAGE_PAIR',):
            fetch_dict[k] = convert_NCHW_to_NHWC(fetch_dict[k])
    
    with h5py.File(prediction_file, 'w') as f:

        number_of_test_iterations = 1 # will be set to the correct value in the while loop
        test_iteration = 0
        while test_iteration < number_of_test_iterations:
            
            data =  session.run(fetch_dict)
            
            # get number of iterations from the info vector
            number_of_test_iterations = int(data['INFO'][0])

            # create group for the current test sample and save the sample id.
            group = f.require_group('snapshot_1/{0}'.format(test_iteration))
            sample_id = (''.join(map(chr, data['SAMPLE_IDS']))).strip()
            group.attrs['sample_id'] = np.string_(sample_id)
            
            # save intrinsics
            group['intrinsics'] = data['INTRINSICS']
            
            # run the network and save outputs for each network iteration 'i'.
            # iteration 0 corresponds to the bootstrap network.
            # we also store the refined depth for each iteration.
            for i in range(4):
                if i == 0:
                    result = bootstrap_net.eval(data['IMAGE_PAIR'], data['image2_2'])      
                else:
                    result = iterative_net.eval(
                        data['IMAGE_PAIR'], 
                        data['image2_2'], 
                        result['predict_depth2'], 
                        result['predict_normal2'], 
                        result['predict_rotation'], 
                        result['predict_translation']
                    )
                # write predictions
                if data_format == 'channels_last':
                    group['predicted_flow/{0}'.format(i)] = result['predict_flow2'][0].transpose([2,0,1])
                    group['predicted_depth/{0}'.format(i)] = result['predict_depth2'][0,:,:,0]
                else:
                    group['predicted_flow/{0}'.format(i)] = result['predict_flow2'][0]
                    group['predicted_depth/{0}'.format(i)] = result['predict_depth2'][0,0]
                    
                predict_motion = np.concatenate((result['predict_rotation'],result['predict_translation']),axis=1)
                group['predicted_motion/{0}'.format(i)] = predict_motion[0]
                
                # run refinement network
                result_refined = refine_net.eval(data['image1'],result['predict_depth2'])
                
                # write refined depth prediction
                if data_format == 'channels_last':
                    group['predicted_depth/{0}_refined'.format(i)] = result_refined['predict_depth0'][0,:,:,0]
                else:
                    group['predicted_depth/{0}_refined'.format(i)] = result_refined['predict_depth0'][0,0]
                
            test_iteration += 1
            
    del session
    tf.reset_default_graph()
    return prediction_file

def main():

    # list the test datasets names for evaluation
    datasets = ('mvs', 'scenes11', 'rgbd', 'sun3d', 'nyu2')
    dataset_dir = os.path.join('..', 'datasets')



    # creating the ground truth and prediction files requires about 11gb of disk space
    for dataset in datasets:
        gt_file = create_ground_truth_file(dataset, dataset_dir)
        
        print('creating predictions for', dataset)
        pr_file = create_prediction_file(dataset, dataset_dir)

        # compute errors
        # the evaluate function expects the path to a prediction and the corresponding
        # ground truth file.
        print('computing errors for', dataset)

        # compute errors for comparison with single image depth methods
        eval_result = evaluate(pr_file, gt_file, depthmask=False, eigen_crop_gt_and_pred=True)
        # save evaluation results to disk
        write_xarray_json(eval_result, '{0}_eval_crop_allpix.json'.format(dataset))
        
        if dataset != 'nyu2':
            # depthmask=True will compute depth errors only for pixels visible in both images.
            eval_result = evaluate(pr_file, gt_file, depthmask=True)
            # save evaluation results to disk
            write_xarray_json(eval_result, '{0}_eval.json'.format(dataset))
            


    # print errors
    for dataset in datasets:
        
        # In the following eval_result is a 5D array with the following dimensions:
        #  - snapshots: stores results of different network training states
        #  - iteration: network iterations '0' stores the result of the bootstrap network.
        #               '3' stores the results after bootstrap + 3 times iterative network.
        #               '3_refined' stores the result after the refinement network.
        #  - sample: the sample number.
        #  - errors: stores the different error metrics.
        #  - scaled: is a boolean dimension used for storing errors after optimal scaling 
        #            the prediction with a scalar factor. This was meant as an alternative
        #            to scale invariant error measures. Just set this to False and ignore.
        # 
        # The following prints the error metrics as used in the paper.

        depth_errors = ['depth_l1_inverse','depth_scale_invariant','depth_abs_relative']
        motion_errors = ['rot_err','tran_angle_err']
        print('======================================')
        print('dataset: ', dataset)
        if dataset != 'nyu2':
            eval_result = read_xarray_json('{0}_eval.json'.format(dataset))
            print('  depth', eval_result[0].loc['3_refined',:,depth_errors,False].mean('sample').to_pandas().to_string())
            print('  motion', eval_result[0].loc['3',:,motion_errors,False].mean('sample').to_pandas().to_string())
        eval_result = read_xarray_json('{0}_eval_crop_allpix.json'.format(dataset))
        print('  depth cropped+all pixels', eval_result[0].loc['3_refined',:,['depth_scale_invariant'],False].mean('sample').to_pandas().to_string())
        

if __name__ == "__main__":
    main()


