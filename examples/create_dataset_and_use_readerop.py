################################################################################
# Create a new dataset h5 file that could be used for training
#
import os
import sys
import numpy as np
from PIL import Image
import h5py

examples_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(examples_dir, '..', 'python'))
from depthmotionnet.dataset_tools import *


# intrinsics supported by DeMoN
normalized_intrinsics = [0.89115971, 1.18821287, 0.5, 0.5]

# unique group name not starting with '.'
group_name = 'sculpture-0001'

# write a new dataset with a single group and two views
with h5py.File('dataset.h5','w') as f:
    
    for i in range(2):
        img = Image.open('sculpture{0}.png'.format(i+1))
        Rt = np.loadtxt('sculpture_Rt{0}.txt'.format(i+1))
        depth = np.load('sculpture_depth{0}.npy'.format(i+1))
        K = np.eye(3)
        K[0,0] = normalized_intrinsics[0] * img.size[0]
        K[1,1] = normalized_intrinsics[1] * img.size[1]
        K[0,2] = normalized_intrinsics[2] * img.size[0]
        K[1,2] = normalized_intrinsics[2] * img.size[1]

        # create a View tuple
        view = View(R=Rt[:,:3], t=Rt[:,3], K=K, image=img, depth=depth, depth_metric='camera_z')

        # write view to the h5 file
        # view enumeration must start with 0 ('v0')
        view_group = f.require_group(group_name+'/frames/t0/v{0}'.format(i))
        write_view(view_group, view)

    # write valid image pair combinations to the group t0
    viewpoint_pairs = np.array([0, 1, 1, 0], dtype=np.int32)
    time_group = f[group_name]['frames/t0']
    time_group.attrs['viewpoint_pairs'] = viewpoint_pairs


################################################################################
# Use the reader op to read the created h5 file
#
from depthmotionnet.datareader import *
import json
import tensorflow as tf
from matplotlib import pyplot as plt


# keys for the requested output tensors. 
# These keys will be passed to the data reader op.
data_tensors_keys = ('IMAGE_PAIR', 'MOTION', 'DEPTH', 'INTRINSICS')

# the following parameters are just an example and are not optimized for training
reader_params = {
     'batch_size': 1,
     'test_phase': False,
     'builder_threads': 1,
     'inverse_depth': True,
     'motion_format': 'ANGLEAXIS6',
     'norm_trans_scale_depth': True,
     # downsampling of image and depth is supported
     'scaled_height': 96,
     'scaled_width': 128,
     'scene_pool_size': 5, # for actual training this should be around 500
     'augment_rot180': 0,
     'augment_mirror_x': 0,
     'top_output': data_tensors_keys, # request data tensors
     'source': [{'path': 'dataset.h5', 'weight': [{'t': 0, 'v': 1.0}]},],
    }

reader_tensors = multi_vi_h5_data_reader(len(data_tensors_keys), json.dumps(reader_params))
# create a dict to make the distinct data tensors accessible via keys
data_dict = dict(zip(data_tensors_keys,reader_tensors[2]))

gpu_options = tf.GPUOptions()
gpu_options.per_process_gpu_memory_fraction=0.8 # leave some memory to other processes
session = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options))

result =  session.run(data_dict)

# show the depth ground truth.
# Note that the data reader op replaces invalid depth values with nan.
plt.imshow(result['DEPTH'].squeeze(), cmap='Greys')
plt.show()

# visualize the data as point cloud if vtk is available
try:
    from depthmotionnet.vis import *
    visualize_prediction(
        inverse_depth=result['DEPTH'], 
        image=result['IMAGE_PAIR'][0,0:3], 
        rotation=result['MOTION'][0,0:3], 
        translation=result['MOTION'][0,3:])
except ImportError as err:
    print("Cannot visualize as pointcloud.", err)



