# DeMoN: Depth and Motion Network

[![License](https://img.shields.io/badge/license-GPLv3-blue.svg)](LICENSE)

DeMoN is "A computer algorithm for reconstructing a scene from two projections"<sup>1</sup>.
The network estimates the depth and relative camera motion for pairs of images; it addresses the important two view case in structure from motion.

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

<sup>1 This is the title of H. C. Longuet-Higgins paper from 1981, which perfectly describes what our method does. DeMoN shows that complex geometric relations can be learnt by a ConvNet.</sup>

## Requirements

Building and using requires the following libraries and programs

    tensorflow 1.4.0
    cmake 3.7.1
    python 3.5
    cuda 8.0.61 (required for gpu support)
    VTK 7.1 with python3 interface (required for visualizing point clouds)

The versions match the configuration we have tested on an ubuntu 16.04 system.
DeMoN can work with other versions of the aforementioned dependencies, e.g. tensorflow 1.3, but this is not well tested.

The binary package from [vtk.org](http://www.vtk.org) does not come with a python3 interface.
To enable python3 support VTK needs to be built from source.
Alternatively, there are also VTK packages with python3 support available in Anaconda via the conda package manager.

The network also depends on our [lmbspecialops](https://github.com/lmb-freiburg/lmbspecialops) library which is included as a submodule.



## Build instructions

The following describes how to install tensorflow and demon into a new virtualenv and run the inference example.
We will use ```pew``` (```pip3 install pew```) to manage a new virtualenv named ```demon_venv``` in the following:

```bash
# create virtualenv
pew new demon_venv
```

The following commands all run inside the virtualenv:

```bash
# install python module dependencies
pip3 install tensorflow-gpu # or 'tensorflow' without gpu support
pip3 install pillow # for reading images
pip3 install matplotlib # required for visualizing depth maps
pip3 install Cython # required for visualizing point clouds
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

