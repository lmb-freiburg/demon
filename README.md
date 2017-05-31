# DeMoN: Depth and Motion Network

[![License](https://img.shields.io/badge/license-GPLv3-blue.svg)](LICENSE)

DeMoN is a ConvNet architecture for solving structure from motion from two views.
It estimates the depth and relative camera motion for pairs of images.

![Teaser](teaser.png)

If you use this code for research please cite:
   
    @InProceedings{UZUMIDB17,
      author       = "B. Ummenhofer and H. Zhou and J. Uhrig and N. Mayer and E. Ilg and A. Dosovitskiy and T. Brox",
      title        = "DeMoN: Depth and Motion Network for Learning Monocular Stereo",
      booktitle    = "IEEE Conference on Computer Vision and Pattern Recognition (CVPR)",
      month        = " ",
      year         = "2017",
      url          = "http://lmb.informatik.uni-freiburg.de//Publications/2017/UZUMIDB17"
    }

See the [project website](https://lmb.informatik.uni-freiburg.de/people/ummenhof/depthmotionnet) for the paper and other material.


## Requirements

Building and using requires the following libraries and programs

    tensorflow 1.0.0
    cmake 3.5.1
    python 3.5
    cuda 8.0.44 (required for gpu support)
    VTK 7.1 with python3 interface (required for visualizing point clouds)

The versions match the configuration we have tested on an ubuntu 16.04 system.

There is no binary package available for VTK with python3 interface and therefore it needs to be built from source.

The network also depends on our [lmbspecialops](https://github.com/lmb-freiburg/lmbspecialops) library which is included as a submodule.



## Build instructions

The following describes how to install tensorflow and demon into a new virtualenv and run the inference example.
We will use ```pew``` (```pip install pew```) to manage a new virtualenv named ```demon_venv``` in the following:

```bash
# create virtualenv
pew new demon_venv
```

The following commands all run inside the virtualenv:

```bash
# install python module dependencies
pip install tensorflow-gpu # or 'tensorflow' without gpu support
pip install pillow # for reading images
pip install matplotlib # required for visualizing depth maps
pip install Cython # required for visualizing point clouds
```

```bash
# clone repo with submodules
git clone --recursive https://github.com/lmb-freiburg/demon.git

# build lmbspecialops
DEMON_DIR=$PWD/demon
mkdir $DEMON_DIR/lmbspecialops/build
cd $DEMON_DIR/lmbspecialops/build
cmake .. # add '-DBUILD_WITH_CUDA=OFF' to build without gpu support
# (optional) run 'ccmake .' here to adjust settings for gpu code generation
make
pew add $DEMON_DIR/lmbspecialops/python # add to python path

# download weights
cd $DEMON_DIR/weights
./download_weights.sh

# run example
cd $DEMON_DIR/examples
python3 example.py # opens a window with the depth map (and the point cloud if vtk is available)
```

## Data reader op & evaluation

The data reader op and the evaluation code have additional dependencies.
The code for the data reader is in the ```multivih5datareaderop``` directory. 
See the corresponding [readme](multivih5datareaderop/README.md) for more details.

For the evaluation see the example [```examples/evaluation.py```](examples/evaluation.py).
The evaluation code requires the following additional python3 packages, which can be installed with ```pip```:

```
h5py
minieigen
pandas
scipy
scikit-image
xarray
```
Note that the evaluation code also depends on the data reader op.



## License

DeMoN is under the [GNU General Public License v3.0](LICENSE.txt)

