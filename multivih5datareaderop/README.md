
# Multi View H5 Data Reader

This document describes the op and the data format used for training DeMoN.

## Building the op

To build the op, create a ```build``` directory inside the demon root directory.
The location of the ```build``` directory is important, because the python package
```depthmotionnet.datareader``` will search this path for the data reader library.

Then run cmake inside the build folder to configure and generate the build
files.
If you use a virtualenv make sure to activate it before running cmake.

Assuming the virtualenv is managed with ```pew``` and named ```demon_venv```
and the demon root directory is stored in the variable ```DEMON_DIR``` we can
build the data reader op with:
```bash
cd $DEMON_DIR # change to the demon root directory

mkdir build 
cd build 

pew in demon_venv
cmake ..
make
```

### Dependencies
The op depends on the following libraries:
```
cmake 3.5.1
tensorflow 1.0.0
hdf5 1.8.16
OpenCV 3.2.0
```
The versions match the configuration we have tested on an ubuntu 16.04 system. 

In addition, the cmake build script will download and build ```lz4```, ```webp```, [```json```](https://github.com/nlohmann/json) and [```half```](http://half.sourceforge.net/)


## `multi_vi_h5_data_reader` Op

```multi_vi_h5_data_reader(num_outputs, param_json)```

Reads data in the multi view h5 format.

This op reads one or more hdf5 files and generates data samples.
The data is returned in NCHW format.

#### Args

**num_outputs**: The number of data tensors to return.  This number depends on the
  values passed in `param_json`.

**param_json**:
  The parameters passed to the reader in JSON format as a string.
  It is recommended to create a python dict with all parameters first and then convert
  the dict to str with json.dumps().
  Here is an example python dict with comments and good values for training:

  ```python
  {
   'batch_size': 32,               # the batch size
   'test_phase': False,            # If True enables testing mode which disables randomization.
   
   # the number of threads used for building batches. For testing set this to 1.
   'builder_threads': 4,           
                                   
   'inverse_depth': True,          # return depth with inverse depth values (1/z)

   # return the motion as one of 'ANGLEAXIS6', 'ANGLEAXIS7' 'QUATERNION', 'FMATRIX'.
   # The translation is stored always in the last 3 elements.
   #
   # ANGLEAXIS6: uses 3 elements for the rotation as angle axis [aa0, aa1, aa2, tx, ty, tz]
   # ANGLEAXIS7: uses 4 elements for the rotation as angle axis [angle, ax, ay, az, tx, ty, tz]
   # QUATERNION: uses 4 elements for the rotation as quaternion [qw, qx, qy, qz, tx, ty, tz]
   # FMATRIX: returns a fundamental matrix in column major order without the last element 
   #          which is defined as 1. [f11, f21, f31, f12, f22, f32, f13, f23]
   'motion_format': 'ANGLEAXIS6', 
   
   # if True normalized the translation ||t||=1 and scales the depth values accordingly.
   'norm_trans_scale_depth': True, 
   
   # the output image/depth height and width.
   # Downsampling is supported. 
   'scaled_height': 192,
   'scaled_width': 256,

   # the number of scenes to keep in memory. A bigger pool improves variance when
   # generating a new batch item, but requires more main memory.
   # For testing a small value like 5 is sufficient.
   'scene_pool_size': 500,

   # The requested output tensors.
   'top_output': ('IMAGE_PAIR', 'MOTION', 'DEPTH', 'INTRINSICS'),

   # probabilities for geometric augmentations.
   # There is a 50% change of rotating the image and cameras by 180 deg followed
   # by a 50% change of mirroring the x-axis.
   # Set this to 0 for testing.
   'augment_rot180': 0.5,
   'augment_mirror_x': 0.5,

   # source is a list of dicts, which define the paths to the hdf5 files and the 
   # importance of each file.
   # In the example below the reader will sample from data2.h5 twice as often as
   # from data1.h5.
   'source': [
              {'path': '/path/to/data1.h5', 'weight': [{'t': 0, 'v': 1.0}]},
              {'path': '/path/to/data2.h5', 'weight': [{'t': 0, 'v': 2.0}]},
             ],
   # for testing only 1 source must be used. Multiple files can be concatenated with ';'.
   #'source': [ {'path': '/path/to/test1.h5;/path/to/test2.h5'}, ],
  }
  ```

#### Outputs

**info**: The info tensor stores information about the internal buffers.
  It stores the following information:
   - required number of test iterations
   - current batch buffer size
   - maximum batch buffer size
   - current reader buffer size
   - maximum reader buffer size

**sample_id**: A tensor storing a string with the id for each batch item.
  A newline symbol is used to separate the individual id strings.

**output**: A list of tensors with the requested data.

  The order of tensors is always:
  ['IMAGE_PAIR', 'MOTION', 'FLOW', 'DEPTH', 'INTRINSICS', 'DEPTHMASKS','SAMPLE_IDS'].
  Depending on the 'top_output' parameter in 'param_json' not all tensors
  may be present.

  The 'IMAGE_PAIR' tensor stores the image pair as 6 channel RGBRGB image.

  The 'MOTION' tensor stores the motion from the first to the second camera in
  the requested format specified by the 'motion_format' parameter in 'param_json'.

  The 'FLOW' tensor stores the optical flow from the first to the second image
  with 2 channels. The first channel stores the x component of the flow vector.

  The 'DEPTH' tensor stores the depth map for the first image.

  The 'INTRINSICS' tensor stores the normalized intrinsics as [fx fy cx cy].
  fx,fy is the x and y component of the normalized focal length.
  cx,cy is the x and y component of the normalized principal point.

  The 'DEPTHMASKS' tensor masks point where it is possible to compute a depth value.



**See also the example [```examples/create_dataset_and_use_readerop.py```](../examples/create_dataset_and_use_readerop.py) for using this op in the examples folder.**



## HDF5 Data Format

Datasets are stored as objects in HDF5 files.
To minimize data IO, we group images that show the same scene. 
A valid group with a unique name "group" stores the following datasets:

```
/group/frames/t0/v0/image
/group/frames/t0/v0/depth
/group/frames/t0/v0/camera
/group/frames/t0/v1/image
/group/frames/t0/v1/depth
/group/frames/t0/v1/camera
...
```

`t0/v0` means viewpoint 0 at time 0. The time is always `t0`. The number of
viewpoints must always be >= 2.
For test datasets the number of viewpoints is always 2.


### Reserved groups
All groups starting with a '.' e.g. `/.config` are reserved and are not treated as data samples.

### `image` dataset

Images are stored in webp format as 1D char arrays.

Attributes:
 * format : scalar string attribute with value `"webp"`



 
### `depth` dataset

Depth maps are stored as half precision floats (16-bit) with LZ4 compression.

Attributes:

 * format       : scalar string attribute with value `"lz4half"`
 * depth_metric : scalar string attribute with value `"camera_z"` or `"ray_length"`
 * extents      : 1D int array with [height, width]
 


### `camera` dataset

The camera dataset stores the intrinsic and extrinsic parameters for the viewpoint.
Camera parameters are stored as 1D double data sets.

Attributes:

 * format : scalar string attribute with value `"pinhole"`

 Interpretation:

`[fx fy skew cx cy r11 r21 r31 r12 r22 r32 r13 r23 r33 tx ty tz]`

The internal parameters fx, fy, cx, cy are compatible with the image dimensions of the image data set


### `t0` group 

The time group `t0` stores an attribute `viewpoint_pairs` which enumerates all
valid image pair combinations.

Attribute: 
  * viewpoint_pairs : 1D int vector. Used by the multiviewh5datareader to generate image pairs. 
  Two subsequent values describe a pair.  E.g. the vector `[0 1 0 2]` describes the pairs (0,1) and (0,2).

For test datasets the value of the `viewpoint_pairs` attribute must be `[0 1]`.



**See also the [```examples/create_dataset_and_use_readerop.py```](../examples/create_dataset_and_use_readerop.py) in the examples folder.**
