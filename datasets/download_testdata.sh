#!/bin/bash
clear
cat << EOF

================================================================================


The test datasets are provided for research purposes only.

Some of the test datasets build upon other publicly available data.
Make sure to cite the respective original source of the data if you use the 
provided files for your research.

  * sun3d_test.h5 is based on the SUN3D dataset http://sun3d.cs.princeton.edu/

    J. Xiao, A. Owens, and A. Torralba, „SUN3D: A Database of Big Spaces Reconstructed Using SfM and Object Labels“, in 2013 IEEE International Conference on Computer Vision (ICCV), 2013, S. 1625–1632.

  

  * rgbd_test.h5 is based on the RGBD SLAM benchmark http://vision.in.tum.de/data/datasets/rgbd-dataset (licensed under CC-BY 3.0)
    
    J. Sturm, N. Engelhard, F. Endres, W. Burgard, and D. Cremers, „A benchmark for the evaluation of RGB-D SLAM systems“, in 2012 IEEE/RSJ International Conference on Intelligent Robots and Systems, 2012, S. 573–580.



  * scenes11_test.h5 uses objects from shapenet https://www.shapenet.org/
    
    A. X. Chang u. a., „ShapeNet: An Information-Rich 3D Model Repository“, arXiv:1512.03012 [cs], Dez. 2015.



  * mvs_test.h5 contains scenes from https://colmap.github.io/datasets.html
    
    J. L. Schönberger und J. M. Frahm, „Structure-from-Motion Revisited“, in 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016, S. 4104–4113.
    J. L. Schönberger, E. Zheng, J.-M. Frahm, und M. Pollefeys, „Pixelwise View Selection for Unstructured Multi-View Stereo“, in Computer Vision – ECCV 2016, 2016, S. 501–518.


================================================================================

type Y to start the download.

EOF

read -s -n 1 answer
if [ "$answer" != "Y" -a "$answer" != "y" ]; then
	exit 0
fi
echo

wget https://lmb.informatik.uni-freiburg.de/data/demon/testdata/sun3d_test.tgz
wget https://lmb.informatik.uni-freiburg.de/data/demon/testdata/rgbd_test.tgz
wget https://lmb.informatik.uni-freiburg.de/data/demon/testdata/mvs_test.tgz
wget https://lmb.informatik.uni-freiburg.de/data/demon/testdata/scenes11_test.tgz
tar -xvf sun3d_test.tgz
tar -xvf rgbd_test.tgz
tar -xvf mvs_test.tgz
tar -xvf scenes11_test.tgz

