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
import os
import math
import itertools
import h5py
import numpy as np
from PIL import Image
from scipy.ndimage import maximum_filter1d, minimum_filter1d

from .view import View
from .view_io import *
from .view_tools import *
from .helpers import measure_sharpness


def read_frameid_timestamp(files):
    """Get frameids and timestamps from the sun3d filenames
    
    files: list of str
        a list of the filenames
        
    Returns the frameid and timestamp as numpy.array
    """
    frameids = []
    timestamps = []
    for f in files:
        id_timestamp = f[:-4].split('-')
        frameids.append( int(id_timestamp[0]) )
        timestamps.append( int(id_timestamp[1]) )
    return np.asarray(frameids), np.asarray(timestamps)


def read_image(filename):
    """Read image from a file
    
    filename: str
    
    Returns image as PIL Image
    """
    image = Image.open(filename)
    image.load()
    return image

def read_depth(filename):
    """Read depth from a sun3d depth file
    
    filename: str
    
    Returns depth as np.float32 array
    """
    depth_pil = Image.open(filename)
    depth_arr = np.array(depth_pil)
    depth_uint16 = depth_arr.astype(np.uint16)
    depth_shifted = (depth_uint16 >> 3) | (depth_uint16 << 13)
    depth_float = (depth_shifted/1000).astype(np.float32)
    return depth_float

def read_Rt(extrinsics, frame):
    """Read camera extrinsics at certain frame
    
    extrinsics: np array with size (totalframe*3, 4)
    
    frame: int (starts from 0)
    
    Returns the rotation and translation 
    """
    Rt = extrinsics[3*frame:3*frame+3]
    R_arr = Rt[0:3,0:3]
    t_arr = Rt[0:3,3]
    R = R_arr.transpose()
    t = -np.dot(R,t_arr)
    return R, t


def compute_sharpness(sun3d_data_path, seq_name):
    """Returns a numpy array with the sharpness score of all images in the sequence.

    sun3d_data_path: str
        base path to the sun3d data

    seq_name: str
        the name of the sequence e.g. "mit_32_d463/d463_1"

    """
    seq_path = os.path.join(sun3d_data_path,seq_name)
    image_files = [f for f in sorted(os.listdir(os.path.join(seq_path,'image'))) if f.endswith('.jpg')]

    sharpness = []
    for img_file in image_files:
        img = read_image(os.path.join(seq_path,'image',img_file))
        sharpness.append(measure_sharpness(img))

    return np.asarray(sharpness)


def create_samples_from_sequence(h5file, sun3d_data_path, seq_name, baseline_range, sharpness, sharpness_window=30, max_views_num=10):
    """Read a sun3d sequence and write samples to the h5file
    
    h5file: h5py.File handle
    
    sun3d_data_path: str
        base path to the sun3d data

    seq_name: str
        the name of the sequence e.g. "mit_32_d463/d463_1"

    baseline_range: tuple(float,float)
        The allowed baseline range

    sharpness: numpy.ndarray 1D
        Array with the sharpness score for each image

    sharpness_window: int
        Window for detecting sharp images

    Returns the number of generated groups
    """
    generated_groups = 0
    seq_path = os.path.join(sun3d_data_path,seq_name)
    group_prefix = seq_name.replace('/','.')
    if not os.path.exists(os.path.join(seq_path, 'extrinsics')):
       return 0

    # file list
    image_files = [f for f in sorted(os.listdir(os.path.join(seq_path,'image'))) if f.endswith('.jpg')]
    depth_files = [f for f in sorted(os.listdir(os.path.join(seq_path,'depthTSDF'))) if f.endswith('.png')]
    extrinsics_files = [f for f in sorted(os.listdir(os.path.join(seq_path,'extrinsics'))) if f.endswith('.txt')]

    # read intrinsics
    intrinsics = np.loadtxt(os.path.join(seq_path,'intrinsics.txt'))

    # read extrinsics params
    extrinsics = np.loadtxt(os.path.join(seq_path,'extrinsics',extrinsics_files[-1]))

    # read time stamp
    img_ids, img_timestamps = read_frameid_timestamp(image_files)
    _, depth_timestamps = read_frameid_timestamp(depth_files)

    # find a depth for each image
    idx_img2depth = []
    for img_timestamp in img_timestamps:
        idx_img2depth.append(np.argmin(abs(depth_timestamps[:] - img_timestamp)))


    # find sharp images with nonmaximum suppression
    assert sharpness.size == len(image_files)
    sharpness_maxfilter = maximum_filter1d(np.asarray(sharpness), size=sharpness_window, mode='constant', cval=0)
    sharp_images_index = np.where( sharpness == sharpness_maxfilter )[0]

    used_views = set()
    for i1, frame_idx1 in enumerate(sharp_images_index):
        if i1 in used_views:
            continue
            
        R1, t1 = read_Rt(extrinsics, frame_idx1)
        i2 = i1+1
        
        depth_file = os.path.join(seq_path,'depthTSDF', depth_files[idx_img2depth[frame_idx1]])
        depth1 = read_depth(depth_file)
        
        if np.count_nonzero(np.isfinite(depth1) & (depth1 > 0)) < 0.5*depth1.size:
            continue
        
        image1 = read_image(os.path.join(seq_path,'image',image_files[frame_idx1]))
        view1 = View(R=R1, t=t1, K=intrinsics, image=image1, depth=depth1, depth_metric='camera_z')
        
        views = [view1]
        used_views.add(i1)
        
        for i2 in range(i1+1, sharp_images_index.size):
            frame_idx2 = sharp_images_index[i2]
            R2, t2 = read_Rt(extrinsics, frame_idx2)
            baseline = np.linalg.norm( (-R1.transpose().dot(t1)) - (-R2.transpose().dot(t2))) # unit is meters
            if baseline < baseline_range[0] or baseline > baseline_range[1]:
                continue
            
            cosine = np.dot(R1[2,:],R2[2,:])
            if cosine < math.cos(math.radians(70)):
                continue
                
            depth_file = os.path.join(seq_path,'depthTSDF', depth_files[idx_img2depth[frame_idx2]])
            depth2 = read_depth(depth_file)
            
            if np.count_nonzero(np.isfinite(depth2) & (depth2 > 0)) < 0.5*depth2.size:
                continue

            view2 = View(R=R2, t=t2, K=intrinsics, image=None, depth=depth2, depth_metric='camera_z')
            check_params = {'min_valid_threshold': 0.4, 'min_depth_consistent': 0.7 }
            if check_depth_consistency(view1, [view2],**check_params) and check_depth_consistency(view2, [view1], **check_params):
                image2 = read_image(os.path.join(seq_path,'image',image_files[frame_idx2]))
                view2 = view2._replace(image=image2)
                views.append(view2)
                used_views.add(i2)
                # print(baseline, cosine)
            if len(views) > max_views_num:
                break
            
        if len(views) > 1:
            group_name = group_prefix+'-{:07d}'.format(img_ids[i1])
            print('writing', group_name)

            view_pairs = []
            for pair in itertools.product(range(len(views)),repeat=2):
                if pair[0] != pair[1]:
                    baseline = np.linalg.norm(views[pair[0]].t-views[pair[1]].t)
                    if baseline >= baseline_range[0] or baseline <= baseline_range[1]:
                        view_pairs.extend(pair)
            for i, v in enumerate(views):
                view_group = h5file.require_group(group_name+'/frames/t0/v{0}'.format(i))
                write_view(view_group, v)

            # write valid image pair combinations to the group t0
            viewpoint_pairs = np.array(view_pairs, dtype=np.int32)
            time_group = h5file[group_name]['frames/t0']
            time_group.attrs['viewpoint_pairs'] = viewpoint_pairs
            generated_groups += 1

    return generated_groups
                
    

