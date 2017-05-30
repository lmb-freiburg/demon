//
//  DeMoN - Depth Motion Network
//  Copyright (C) 2017  Benjamin Ummenhofer, Huizhong Zhou
//  
//  This program is free software: you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation, either version 3 of the License, or
//  (at your option) any later version.
//  
//  This program is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//  
//  You should have received a copy of the GNU General Public License
//  along with this program.  If not, see <http://www.gnu.org/licenses/>.
//
#include "multivih5datareader.h"

#include "simpleh5file.h"

#include <chrono>
#include <random>
#include <thread>
#include <atomic>
#include <mutex>
#include <queue>
#include <algorithm>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include "lz4.h"
#include "webp/decode.h"
#include "half.hpp"


namespace multiviewh5datareader_internal
{
  // maximum number of batches that are kept ready for use in memory
  const int MAX_PRELOADED_BATCHES = 4;

  // blob with same interface as caffe::Blob
  template <class Dtype>
  class MyBlob
  {
  public:
    MyBlob()
      :count_(0)
    { }

    explicit MyBlob(const int num, const int channels, const int height, const int width)
    {
      Reshape(num, channels, height, width);
    }
    explicit MyBlob(const std::vector<int>& shape)
    {
      Reshape(shape);
    }

    void Reshape(const int num, const int channels, const int height, const int width)
    {
      std::vector<int> shape{num, channels, height, width};
      Reshape(shape);
    }
    void Reshape(const std::vector<int>& shape)
    {
      if( shape == shape_ )
        return;
      shape_ = shape;
      count_ = 1;
      for( int s : shape )
        count_ *= s;
      data_.resize(count_);
    }

    inline const std::vector<int>& shape() const
    { 
      return shape_;
    }
    inline int shape(int index) const
    {
      return shape_[CanonicalAxisIndex(index)];
    }

    inline int num_axes() const
    { 
      return shape_.size();
    }

    inline int count() const
    { 
      return count_;
    }

    inline int count(int start_axis, int end_axis) const
    {
      assert(start_axis <= end_axis);
      assert(start_axis >= 0);
      assert(end_axis >= 0);
      assert(start_axis <= num_axes());
      assert(end_axis <= num_axes());
      int count = 1;
      for(int i = start_axis; i < end_axis; ++i)
      {
        count *= shape(i);
      }
      return count;
    }
    inline int count(int start_axis) const
    {
      return count(start_axis, num_axes());
    }

    inline int CanonicalAxisIndex(int axis_index) const
    {
      assert(axis_index >= -num_axes());
      assert(axis_index < num_axes());
      if(axis_index < 0) 
      {
        return axis_index + num_axes();
      }
      return axis_index;
    }

    inline int num() const { return LegacyShape(0); }
    inline int channels() const { return LegacyShape(1); }
    inline int height() const { return LegacyShape(2); }
    inline int width() const { return LegacyShape(3); }
    inline int LegacyShape(int index) const {
      assert(num_axes() <= 4);
      assert(index < 4);
      assert(index >= -4);
      if(index >= num_axes() || index < -num_axes())
      {
        return 1;
      }
      return shape(index);
    }

    inline int offset(const int n, const int c = 0, const int h = 0, const int w = 0) const 
    {
      assert(n>=0 && n<num());
      assert(c>=0 && c<channels());
      assert(h>=0 && h<height());
      assert(w>=0 && w<width());
      return ((n * channels() + c) * height() + h) * width() + w;
    }

    inline int offset(const std::vector<int>& indices) const 
    {
      assert(indices.size() <= num_axes());
      int offset = 0;
      for(int i = 0; i < num_axes(); ++i)
      {
        offset *= shape(i);
        if (indices.size() > i)
        {
          assert(indices[i] >= 0);
          assert(indices[i] < shape(i));
          offset += indices[i];
        }
      }
      return offset;
    }

    inline Dtype data_at(const int n, const int c, const int h, const int w) const
    {
      return cpu_data()[offset(n, c, h, w)];
    }

    inline Dtype data_at(const std::vector<int>& index) const
    {
      return cpu_data()[offset(index)];
    }

    const Dtype* cpu_data() const
    { 
      return &data_[0]; 
    }
    Dtype* mutable_cpu_data()
    {
      return &data_[0];
    }
    
    size_t byte_size() const
    {
      return count_*sizeof(Dtype);
    }

  private:
    std::vector<Dtype> data_, diff_;
    std::vector<int> shape_;
    int count_;
  };

  // indices for storing the order of tops.
  // -1 means no top for this data.
  struct TopIndices
  {
    TopIndices()
      :images(-1),depths(-1), flow(-1), motion(-1), intrinsics(-1), depthmasks(-1)
    { }
    int images;
    int depths;
    int flow;
    int motion;
    int intrinsics;
    int depthmasks;
  };

  struct Camera
  {
    std::string image_data; 
    int width, height, channels;

    enum DepthMode { CAMERA_Z=0, RAY_LENGTH=1 };
    int depth_mode;
    std::string depth_data;

    Eigen::Matrix3d K;
    Eigen::Matrix3d R;
    Eigen::Vector3d t;
  };

  typedef std::vector<Camera> CameraVec;

  struct Scene
  {
    // each camera can have multiple sub cameras which are stored in a CameraVec
    // cameras[0][1] is the second subcamera of the first camera 
    std::vector<CameraVec> cameras;
    // a list of allowed image pair combinations
    std::vector<std::pair<int,int>> combinations;
    uint64_t seed;

    std::string scene_id;
    int source_id;
  };
  typedef std::shared_ptr<Scene> Scene_sptr;


  // splits the string using the delimiter
  std::vector<std::string> split( const std::string& str, const char delimiter )
  {
    std::istringstream isstr(str);
    std::vector<std::string> result;
    std::string item;
    while( std::getline(isstr, item, delimiter) )
      result.push_back(item);
    return result;
  }


  template <class Derived>
  void rotationMatrixToAngleAxis( double* aa, const Eigen::MatrixBase<Derived>& R )
  {
    EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(Derived, 3, 3);
    Eigen::AngleAxisd angle_axis(R);
    Eigen::Map<Eigen::Vector3d> out(aa);
    out = angle_axis.axis()*angle_axis.angle();
  }

 
  template <class T>
  Eigen::Matrix<T,3,3> computeFundamentalFromCameras( 
      const Eigen::Matrix<T,3,4>& P1, 
      const Eigen::Matrix<T,3,4>& P2 )
  {
    Eigen::Matrix<T,3,3> F;
    Eigen::Matrix<T,2,4> X1, X2, X3;
    X1 = P1.bottomRows(2);
    X2.topRows(1) = P1.bottomRows(1);
    X2.bottomRows(1) = P1.topRows(1);
    X3 = P1.topRows(2);

    Eigen::Matrix<T,2,4> Y1, Y2, Y3;
    Y1 = P2.bottomRows(2);
    Y2.topRows(1) = P2.bottomRows(1);
    Y2.bottomRows(1) = P2.topRows(1);
    Y3 = P2.topRows(2);

    Eigen::Matrix<T,4,4> tmp;
    tmp << X1, Y1;
    F(0,0) = tmp.determinant();
    tmp << X2, Y1;
    F(0,1) = tmp.determinant();
    tmp << X3, Y1;
    F(0,2) = tmp.determinant();
    
    tmp << X1, Y2;
    F(1,0) = tmp.determinant();
    tmp << X2, Y2;
    F(1,1) = tmp.determinant();
    tmp << X3, Y2;
    F(1,2) = tmp.determinant();
    
    tmp << X1, Y3;
    F(2,0) = tmp.determinant();
    tmp << X2, Y3;
    F(2,1) = tmp.determinant();
    tmp << X3, Y3;
    F(2,2) = tmp.determinant();
    
    return F;
  }

  template <class T>
  void rotateCamera180DegAroundZ(Eigen::Matrix<T,3,3>& R, Eigen::Matrix<T,3,1>& t)
  {
    Eigen::Matrix<T,3,1> C = - R.transpose()*t;
    R.row(0) = -R.row(0);
    R.row(1) = -R.row(1);
    t = -R*C;
  }

  // convert an image in bgr format to a gray value float image with range 
  // [range_min..range_max]
  template <class T>
  void convertBGRToFloatGray(std::string& image, T range_min, T range_max)
  {
    if( image.size()%3 )
      throw std::runtime_error("Conversion to gray value image failed");
    const int num_elements = image.size()/3;
    std::string tmp;
    tmp.resize(sizeof(T)*num_elements);

    const T range = range_max-range_min;

    uint8_t *ptr = (uint8_t*)&image[0];
    T *tmp_ptr = (T*)&tmp[0];
    T scale_b = range/255*5/32;
    T scale_g = range/255*16/32;
    T scale_r = range/255*11/32;
    for( int i = 0; i < num_elements; ++i, ptr+=3 )
    {
      tmp_ptr[i] = scale_b*ptr[0] + scale_g*ptr[1] + scale_r*ptr[2];
      tmp_ptr[i] += range_min;
    }
    image = tmp;
  }

  // convert an image in bgr format to an image with 3 separate color layers.
  // each layer uses the float range [range_min..range_max]
  template <class T>
  void convertBGRToFloatRGBLayers(std::string& image, int width, int height, T range_min, T range_max)
  {
    if( (int)image.size() != width*height*3 )
      throw std::runtime_error("Conversion to rgb layer image failed value image failed");
    std::string tmp;
    tmp.resize(sizeof(T)*image.size());

    const T range = range_max-range_min;

    uint8_t *ptr = (uint8_t*)&image[0];
    T *tmp_ptr = (T*)&tmp[0];
    T scale = range/255;
    int size_xy = width*height;
    for( int y = 0; y < height; ++y )
    for( int x = 0; x < width; ++x, ptr+=3 )
    {
      tmp_ptr[0*size_xy + y*width + x] = scale*ptr[2] + range_min;
      tmp_ptr[1*size_xy + y*width + x] = scale*ptr[1] + range_min;
      tmp_ptr[2*size_xy + y*width + x] = scale*ptr[0] + range_min;
    }
    image = tmp;
  }

  // computes the flow for the image pair cam1, cam2
  // The flow describes the motion from the image of cam1 to the image of cam2
  template <class T>
  void computeFlow( T* flow, const Camera& cam1, const Camera& cam2 )
  {
    Eigen::Matrix3f K2 = cam2.K.cast<float>();
    K2(0,0) *= cam2.width;
    K2(1,1) *= cam2.height;
    K2(0,2) *= cam2.width;
    K2(1,2) *= cam2.height;
    Eigen::Matrix<float,3,4> P2;
    P2 << cam2.R.cast<float>(), cam2.t.cast<float>();
    P2 = K2 * P2;
    Eigen::Matrix3f K1 = cam1.K.cast<float>();
    K1(0,0) *= cam1.width;
    K1(1,1) *= cam1.height;
    K1(0,2) *= cam1.width;
    K1(1,2) *= cam1.height;
    Eigen::Matrix3f inv_K = K1.inverse();
    Eigen::Vector3f t = cam1.t.cast<float>();
    Eigen::Matrix3f inv_R = cam1.R.transpose().cast<float>();

    const float* depth_data = (const float*) &cam1.depth_data[0];
    const int xy_size = cam1.width*cam1.height;
    for( int y = 0; y < cam1.height; ++y )
    for( int x = 0; x < cam1.width; ++x, ++depth_data, ++flow )
    {
      Eigen::Vector2f p1(x+0.5f,y+0.5f);
      Eigen::Vector3f pos;
      pos.x() = inv_K(0,0)*p1.x() + inv_K(0,2);
      pos.y() = inv_K(1,1)*p1.y() + inv_K(1,2);
      pos.z() = 1;

      float depth = *depth_data;
      if( depth <= 0 || !std::isfinite(depth) )
      {
        flow[0] = NAN;
        flow[xy_size] = NAN;
        continue;
      }

      float norm = 1;
      if( cam1.depth_mode == Camera::RAY_LENGTH )
        norm = pos.norm();

      pos *= depth/norm;
      pos -= t;
      pos = inv_R*pos;

      Eigen::Vector3f p2 = P2*pos.homogeneous();
      p2.x() /= p2.z();
      p2.y() /= p2.z();

      flow[0] = p2.x()-p1.x();
      flow[xy_size] = p2.y()-p1.y();
    }

  }


  // computes the binary mask where the depth can be computed from the stereo pair.
  // To be able to compute the depth, the same point and a small neighbourhood 
  // must be visible in both images.
  template <class T>
  void computeDepthmask( T* mask, const Camera& cam1, const Camera& cam2, const int border1, const int border2 )
  {
    Eigen::Matrix3f K2 = cam2.K.cast<float>();
    K2(0,0) *= cam2.width;
    K2(1,1) *= cam2.height;
    K2(0,2) *= cam2.width;
    K2(1,2) *= cam2.height;
    Eigen::Matrix<float,3,4> P2;
    P2 << cam2.R.cast<float>(), cam2.t.cast<float>();
    P2 = K2 * P2;
    Eigen::Matrix3f K1 = cam1.K.cast<float>();
    K1(0,0) *= cam1.width;
    K1(1,1) *= cam1.height;
    K1(0,2) *= cam1.width;
    K1(1,2) *= cam1.height;
    Eigen::Matrix3f inv_K = K1.inverse();
    Eigen::Vector3f t = cam1.t.cast<float>();
    Eigen::Matrix3f inv_R = cam1.R.transpose().cast<float>();

    const float* depth_data = (const float*) &cam1.depth_data[0];
    for( int y = 0; y < cam1.height; ++y )
    for( int x = 0; x < cam1.width; ++x, ++depth_data, ++mask )
    {
      if( x < border1 || y < border1 || 
          x >= cam1.width-border1 || y >= cam1.height-border1 )
      {
        mask[0] = 0;
        continue;
      }

      Eigen::Vector2f p1(x+0.5f,y+0.5f);
      Eigen::Vector3f pos;
      pos.x() = inv_K(0,0)*p1.x() + inv_K(0,2);
      pos.y() = inv_K(1,1)*p1.y() + inv_K(1,2);
      pos.z() = 1;

      float norm = 1;
      if( cam1.depth_mode == Camera::RAY_LENGTH )
        norm = pos.norm();

      float depth = *depth_data;
      if( depth <= 0 || !std::isfinite(depth))
      {
        mask[0] = 0;
        continue;
      }

      pos *= depth/norm;
      pos -= t;
      pos = inv_R*pos;

      Eigen::Vector3f p2 = P2*pos.homogeneous();
      p2.x() /= p2.z();
      p2.y() /= p2.z();

      if( p2.x() < border2 || p2.y() < border2 || 
          p2.x() >= cam1.width-border2 || p2.y() >= cam1.height-border2 )
      {
        mask[0] = 0;
      }
      else
      {
        mask[0] = 1;
      }

    }

  }

  template <class Dtype>
  void mirrorImageX( MyBlob<Dtype>& image, int batch_idx )
  {
    if( image.num_axes() != 4 )
      throw std::runtime_error("image blob is not a batch of images");

    int size_x = image.shape(-1);
    int size_y = image.shape(-2);
    int size_z = image.shape(-3);
    int size_xy = size_x*size_y;

    Dtype* data = image.mutable_cpu_data() + image.offset(batch_idx);
    for( int z = 0; z < size_z; ++z )
    {
      for( int y = 0; y < size_y; ++y )
      {
        Dtype* data_fwd = data + z*size_xy + y*size_x;
        Dtype* data_bwd = data + z*size_xy + (y+1)*size_x;
        std::reverse(data_fwd, data_bwd);
      }
    }
  }

  template <class Dtype>
  void rotateImageBy180( MyBlob<Dtype>& image, int batch_idx )
  {
    if( image.num_axes() < 3 )
      throw std::runtime_error("image blob is not a batch of images");

    int size_xy = image.shape(-1)*image.shape(-2);
    int size_z = image.count(1,image.num_axes()-2);

    Dtype* data = image.mutable_cpu_data() + image.offset(batch_idx);
    for( int z = 0; z < size_z; ++z )
    {
      Dtype* data_fwd = data + z*size_xy;
      Dtype* data_bwd = data + (z+1)*size_xy;
      std::reverse(data_fwd, data_bwd);
    }
  }

  template <class T>
  T saturate( T value )
  {
    return std::max(T(0), std::min(T(1),value));
  }

  void rgb2hsv( float& h, float& s, float& v, float r, float g, float b)
  {
    float min_rgb = std::min(r,std::min(g,b));
    v = std::max(r,std::max(g,b));
    
    if( r == v )
      h = 60*(g-b)/(v-min_rgb+1e-6f);
    else if( g == v )
      h = 120+60*(r-g)/(v-min_rgb+1e-6f);
    else
      h = 240+60*(r-g)/(v-min_rgb+1e-6f);
    s = (v-min_rgb)/(v+1e-6f);
  }

  void hsv2rgb( float& r, float& g, float& b, float h, float s, float v )
  {
    int i;
    float f, p, q, t;

    if( s == 0 ) {
      // achromatic (grey)
      r = g = b = v;
      return;
    }

    h /= 60;      // sector 0 to 5
    i = floor( h );
    f = h - i;      // factorial part of h
    p = v * ( 1 - s );
    q = v * ( 1 - s * f );
    t = v * ( 1 - s * ( 1 - f ) );

    switch( i ) {
      case 0:
        r = v;
        g = t;
        b = p;
        break;
      case 1:
        r = q;
        g = v;
        b = p;
        break;
      case 2:
        r = p;
        g = v;
        b = t;
        break;
      case 3:
        r = p;
        g = q;
        b = v;
        break;
      case 4:
        r = t;
        g = p;
        b = v;
        break;
      default:    // case 5:
        r = v;
        g = p;
        b = q;
        break;
    }

  }


  template <class T> 
  T getRandomParam(
      const MultiViH5Params::Source::RandomParam& param,
      std::mt19937& rng )
  {
    if( param.normal.is_valid() )
    {
      std::normal_distribution<T> dist(param.normal->mean, param.normal->stddev);
      return dist(rng);
    }
    else if( param.uniform.is_valid() )
    {
      std::uniform_real_distribution<T> dist(param.uniform->a, param.uniform->b);
      return dist(rng);
    }
    return 0;
  }

  float fast_powf(float a, float b) 
  {
	  union { float d; int x; } u = { a };
	  u.x = (int)(b * (u.x - 1064866805) + 1064866805);
	  return u.d;
  }

  template <class Dtype>
  void augmentImage( 
      MyBlob<Dtype>& image, int batch_idx, 
      const MultiViH5Params::Source& param,
      std::mt19937& rng )
  {
    assert( image.shape().size() == 4 );
    assert( image.shape(1) == 6 ); // assume 2 color images
    const int width = image.shape(-1);
    const int height = image.shape(-2);
    const int step_ch = image.offset(0,1) - image.offset(0);
    const int step_row = image.offset(0,0,1) - image.offset(0);
    
    if( param.aug_hsv_hue.is_valid() || param.aug_hsv_sat.is_valid() || param.aug_hsv_val.is_valid() ||
        param.aug_contrast.is_valid() || param.aug_brightness.is_valid() || param.aug_gamma.is_valid() )
    {
      float hue_change = 0; 
      float val_change = 0;
      float sat_change = 0;
      if( param.aug_hsv_hue.is_valid() )
        hue_change = getRandomParam<float>(param.aug_hsv_hue.value(),rng);
      if( param.aug_hsv_sat.is_valid() )
        sat_change = getRandomParam<float>(param.aug_hsv_sat.value(),rng);
      if( param.aug_hsv_val.is_valid() )
        val_change = getRandomParam<float>(param.aug_hsv_val.value(),rng);

      float contrast = 1;
      float brightness = 0;
      float gamma = 1;
      if( param.aug_contrast.is_valid() )
        contrast = getRandomParam<float>(param.aug_contrast.value(),rng);
      if( param.aug_brightness.is_valid() )
        brightness = getRandomParam<float>(param.aug_brightness.value(),rng);
      if( param.aug_gamma.is_valid() )
        gamma = getRandomParam<float>(param.aug_gamma.value(),rng);
      
      for( int image_i = 0; image_i < 2; ++image_i )
      {
        Dtype* data = image.mutable_cpu_data()+image.offset(batch_idx,3*image_i);

        for( int y = 0; y < height; ++y )
        for( int x = 0; x < width; ++x )
        {
          float rgb[3]; // value range is [0,1]
          rgb[0] = data[2*step_ch+y*step_row+x]+0.5f;
          rgb[1] = data[1*step_ch+y*step_row+x]+0.5f;
          rgb[2] = data[0*step_ch+y*step_row+x]+0.5f;
          float h,s,v;
          rgb2hsv(h, s, v, rgb[0], rgb[1], rgb[2]);

          h += hue_change;
          while( h < 0 ) h += 360.f;
          while( h >= 360 ) h -= 360.f;

          s = saturate(s+sat_change);
          v = saturate(v+val_change);
          hsv2rgb(rgb[0], rgb[1], rgb[2], h, s, v);

          for( int i = 0; i < 3; ++i )
          {
            float value = rgb[i];
            value = (value-0.5f)*contrast + brightness + 0.5f;
            value = fast_powf(value, gamma);
            value = saturate(value);
            rgb[i] = value;
          }

          data[2*step_ch+y*step_row+x] = rgb[0] - 0.5f;
          data[1*step_ch+y*step_row+x] = rgb[1] - 0.5f;
          data[0*step_ch+y*step_row+x] = rgb[2] - 0.5f;
        }
      }
    }

  }


  // Counts the number of groups that do not start with a '.' in an hdf5 file
  size_t countSampleGroups( const std::string& h5_path )
  {
    if( !SimpleH5File::isHDF5(h5_path) )
      throw std::runtime_error("cannot open '" + h5_path + "'");
    SimpleH5File h5file(h5_path, SimpleH5File::READ);
    std::vector<std::string> root_groups = h5file.listGroups("/");
    size_t count = 0;
    for( const std::string g : root_groups )
      if( g[0] != '.' )
        ++count;

    return count;
  }

  size_t countSampleGroups( const std::vector<std::string>& h5_paths )
  {
    size_t count = 0;
    for( const std::string path : h5_paths )
      count += countSampleGroups(path);
    return count;
  }

  //
  // A source reads raw data from a list of hdf5 files.
  //
  struct Source
  {
    Source( 
        const MultiViH5Params::Source& param, 
        TopIndices top_idx, bool test_phase, int source_id)
      :top_idx(top_idx), test_phase(test_phase), source_id(source_id)
    {
      // parse weights
      if( param.weight.size() )
      {
        for( size_t i = 0; i < param.weight.size(); ++i )
        {
          KeyFrame kf;
          kf.time = param.weight[i].t;
          kf.value = param.weight[i].v;
          if( kf.value < 0 )
          {
            std::stringstream errstr;
            errstr << "negative value for keyframe is not allowed, "
              << "(" << kf.time << ", " << kf.value << ")\n";
            throw std::runtime_error(errstr.str());
          }
          keyframes.push_back(kf);
        }
        std::sort(keyframes.begin(), keyframes.end());
        auto comp_fn = [](const KeyFrame& a, const KeyFrame& b){ return a.time == b.time; };
        auto it = std::adjacent_find(keyframes.begin(), keyframes.end(), comp_fn);
        if( it != keyframes.end() )
        {
          std::stringstream errstr;
          errstr << "duplicate keyframe found (" << it->time << ", " << it->value << ")\n";
          throw std::runtime_error(errstr.str());
        }
      }
      else
      {
        keyframes.push_back(KeyFrame{0,1.f});
      }
      

      h5_paths = split(param.path, ';');
      if( h5_paths.empty() )
        throw std::runtime_error("No h5 files specified!");
      // check all the databases 
      bool all_paths_ok = true;
      for( std::string path : h5_paths )
        all_paths_ok = all_paths_ok && SimpleH5File::isHDF5(path);
      if( !all_paths_ok )
        throw std::runtime_error("some h5 files cannot be opened!");

      // shuffle file order in train phase
      if( !test_phase )
        std::random_shuffle(h5_paths.begin(),h5_paths.end());

      // open first h5 file
      current_h5_idx = -1;
      advance();
    }


    void advance()
    {
      current_group = "";
      while( current_group.empty() )
      {
        if( root_groups.empty() )
        {
          current_h5_idx = (current_h5_idx+1) % h5_paths.size();
          if( h5_paths.size() > 1 || !h5file.isOpen() )
          {
            // switch to next file
            h5file.close();
            std::cerr << "opening " << h5_paths[current_h5_idx] << std::endl;
            h5file.open(h5_paths[current_h5_idx], SimpleH5File::READ);
          }
          root_groups = h5file.listGroups("/");
          if( test_phase )
            // do not rely on the group order from the hdf5 lib
            std::sort(root_groups.begin(), root_groups.end());
          else
            std::random_shuffle(root_groups.begin(), root_groups.end());

        }
        // goto next group that does not start with '.'
        while( root_groups.size() )
        {
          std::string tmp = root_groups.back();
          root_groups.pop_back();
          if( tmp[0] != '.' )
          {
            current_group = tmp;
            break;
          }
        }
      }
    }

    Scene_sptr create_scene()
    {
      const int MAX_VIEWPOINTS = 6;

      Scene_sptr scene(new Scene());
      scene->source_id = source_id;
      scene->scene_id = current_group;

      if( h5file.existsAttribute("seed", "/"+current_group) )
        h5file.readAttribute( scene->seed, "seed", "/"+current_group );
      else
        scene->seed = std::numeric_limits<uint64_t>::max();

      std::string t0_group = "/" + current_group + "/frames/t0";
      std::vector<std::string> viewpoint_groups = h5file.listGroups(t0_group);
      if( viewpoint_groups.size() < 2 )
      {
        std::stringstream errstr;
        errstr << "not enough viewpoints in " 
               << h5_paths[current_h5_idx] << t0_group << std::endl;
        throw std::runtime_error(errstr.str());
      }

      // generate image pair combinations
      std::set<int> viewpoint_ids; // the viewpoint ids that we need to load
      std::vector<std::pair<int,int>> pairs;
      if( h5file.existsAttribute("viewpoint_pairs", t0_group) )
      {
        std::vector<int> tmp_pairs;
        h5file.readAttribute(tmp_pairs, "viewpoint_pairs", t0_group);
        if( test_phase ) // limit to the first image pair in test phase
          pairs.resize(1);
        else
          pairs.resize(tmp_pairs.size()/2);
        for( size_t i = 0; i < pairs.size(); ++i )
        {
          std::pair<int,int> p(tmp_pairs[i*2+0], tmp_pairs[i*2+1]);
          pairs[i] = p;
        }
      }
      else if( test_phase )
      {
        pairs.push_back(std::make_pair(0,1));
      }
      else // generate all possible image pairs
      {
        int viewpoint_count = viewpoint_groups.size();
        for( int i1 = 0; i1 < viewpoint_count; ++i1 )
        for( int i2 = 0; i2 < viewpoint_count; ++i2 )
        {
          if( i1 == i2 )
            continue;
          pairs.push_back(std::make_pair(i1,i2));
        }
      }
      // shuffle pairs and then add combinations until we have reached the
      // maximum number of viewpoints (MAX_VIEWPOINTS) or all combinations
      // have been added
      if( !test_phase )
        std::random_shuffle(pairs.begin(), pairs.end());
      for( auto iter = pairs.begin(); iter != pairs.end() && viewpoint_ids.size() < MAX_VIEWPOINTS; ++iter )
      {
        int new_vp_count = int(viewpoint_ids.count(iter->first) == 0);
        new_vp_count += int(viewpoint_ids.count(iter->second) == 0);
        if( new_vp_count + viewpoint_ids.size() <= MAX_VIEWPOINTS )
        {
          viewpoint_ids.insert(iter->first);
          viewpoint_ids.insert(iter->second);
          scene->combinations.push_back(*iter);
        }
      }

      // remap scene->combinations from viewpoint indices to the scene->camera vector indices
      {
        int i = 0;
        std::map<int,int> vpid_camid_map;
        for( auto iter = viewpoint_ids.begin(); iter != viewpoint_ids.end(); ++iter,++i)
          vpid_camid_map[*iter] = i;
        for( std::pair<int,int>& p : scene->combinations )
        {
          p.first = vpid_camid_map[p.first];
          p.second = vpid_camid_map[p.second];
        }
      }

      scene->cameras.resize(viewpoint_ids.size());
      {
        int cam_i = 0;
        for( auto iter = viewpoint_ids.begin(); iter != viewpoint_ids.end(); ++iter,++cam_i)
        {

          std::stringstream group;
          group << t0_group << "/v" << *iter;
          int sub_views = 1;
          if( h5file.existsAttribute("sub_views", group.str()) )
            h5file.readAttribute(sub_views, "sub_views", group.str());

          CameraVec& camvec = scene->cameras[cam_i];

          for( int subcam_idx = 0; subcam_idx < sub_views; ++subcam_idx )
          {
            Camera cam;

            // image
            if( top_idx.images >= 0 || top_idx.intrinsics >= 0 )
            {
              std::string ds_path = group.str() + "/image";
              if(sub_views > 1)
                ds_path += "/" + std::to_string(subcam_idx);
              std::string attr;
              h5file.readAttribute(attr, "format", ds_path);
              if( attr != "webp" )
              {
                throw std::runtime_error("wrong format '" + attr + "' in " + ds_path);
              }
              cam.image_data.resize(h5file.getDatasetExtents(ds_path)[0]);
              h5file.readDataset( &cam.image_data[0], ds_path );

              // read image extents
              WebPGetInfo((uint8_t*)&cam.image_data[0], cam.image_data.size(), &cam.width, &cam.height);
            }

            // depth
            if( top_idx.depths >= 0 || top_idx.flow >= 0 || top_idx.depthmasks)
            {
              std::string ds_path = group.str() + "/depth";
              if(sub_views > 1)
                ds_path += "/" + std::to_string(subcam_idx);
              std::string attr;
              h5file.readAttribute(attr, "format", ds_path);
              if( attr != "lz4half" )
              {
                throw std::runtime_error("wrong format '" + attr + "' in " + ds_path);
              }
              h5file.readAttribute(attr, "depth_metric", ds_path);
              if( attr == "camera_z" )
                cam.depth_mode = Camera::CAMERA_Z;
              else if( attr == "ray_length" )
                cam.depth_mode = Camera::RAY_LENGTH;
              else
                throw std::runtime_error("unknown depth metric '" + attr + " in " + ds_path);

              cam.depth_data.resize(h5file.getDatasetExtents(ds_path)[0]);
              h5file.readDataset( &cam.depth_data[0], ds_path );

              // read extents
              if( top_idx.images == -1 )
              {
                std::vector<int> extents;
                h5file.readAttribute(extents, "extents", ds_path);
                cam.height = extents[0];
                cam.width = extents[1];
              }
            }
          
            // camera
            {
              std::string ds_path = group.str() + "/camera";
              std::string attr;
              h5file.readAttribute(attr, "format", ds_path);
              if( attr != "pinhole" )
              {
                throw std::runtime_error("wrong format '" + attr + "' in " + ds_path);
              }
              std::vector<double> cam_params(17);
              h5file.readDataset( &cam_params[0], ds_path );
              cam.K.setZero();
              cam.K(0,0) = cam_params[0]; // fx
              cam.K(1,1) = cam_params[1]; // fy
              cam.K(0,1) = cam_params[2]; // skew
              cam.K(0,2) = cam_params[3]; // cx
              cam.K(1,2) = cam_params[4]; // cy
              cam.K(2,2) = 1;
              int lin_idx = 5;
              for( int col = 0; col < 3; ++col )
              for( int row = 0; row < 3; ++row, ++lin_idx )
                cam.R(row,col) = cam_params[lin_idx];
              cam.t.x() = cam_params[14];
              cam.t.y() = cam_params[15];
              cam.t.z() = cam_params[16];
            }
            camvec.push_back(cam);
          } // subcam_idx
        } // viewpoint_ids iter
      }


      return scene;
    }

    // returns the interpolated weight of this source for the given iteration
    float weight( int iteration )
    {
      std::vector<KeyFrame>::iterator it1, it2;
      it2 = std::upper_bound(keyframes.begin(), keyframes.end(), KeyFrame{iteration,0.f});
      if( it2 != keyframes.begin() )
        it1 = it2-1;
      else
        it1 = it2;

      if( it1 != it2 )
      {
        if( it2 != keyframes.end() )
        {
          float t = (iteration - it1->time) / float(it2->time - it1->time);
          return (1-t)*it1->value + t*it2->value;
        }
        else
        {
          return it1->value;
        }
      }
      else
      {
        return it1->value;
      }
    }

    struct KeyFrame
    {
      int time;
      float value;
      bool operator<(const KeyFrame& b) const { return time < b.time; }
    };
    std::vector<KeyFrame> keyframes;

    TopIndices top_idx;
    bool test_phase;
    int source_id;

    std::vector<std::string> h5_paths;
    int current_h5_idx;
    std::vector<std::string> root_groups;
    std::string current_group;

    SimpleH5File h5file;
  };


  //
  // thread that just reads data from sources.
  // no decompression, no data processing, just reading
  //
  class DataReader
  {
  public:
    DataReader( 
        const MultiViH5Params& param, 
        TopIndices top_idx, 
        std::atomic<int>* iteration )
      :param(param),
      iteration(iteration),
      max_buffer_size(param.batch_size), 
      quit(false)
    {
      for( size_t i = 0; i < param.source.size(); ++i )
      {
        sources.push_back( std::unique_ptr<Source>(
              new Source( param.source[i], 
                          top_idx, 
                          param.test_phase,
                          i)));
      }

      // the sum of weights must be > 0 for all iterations
      std::vector<Source::KeyFrame> all_keyframes;
      for( auto& s : sources )
        all_keyframes.insert(all_keyframes.end(), s->keyframes.begin(), s->keyframes.end());
      
      for( Source::KeyFrame& kf : all_keyframes )
      {
        double weights = 0;
        for( auto& s : sources )
          weights += s->weight(kf.time);
        if( weights <= 0 )
          throw std::runtime_error("sum of weights must be > 0 for all iterations");
      }

    }

    // no copy
    DataReader( const DataReader& ) = delete;


    void start()
    {
      thread = std::thread(&DataReader::run, this);
    }

    void stop()
    {
      quit = true;
      if( thread.joinable() )
          thread.join();
    }

    Scene_sptr getData()
    {
      Scene_sptr result;
      while( !result )
      {
        std::lock_guard<std::mutex> lock(buffer_mutex);
        //LOG_IF(INFO,buffer.size()>1) << "data reader buffer size " << buffer.size();
        if( buffer.size() )
        {
          result = buffer.front();
          buffer.pop();
        }
        //else
          //LOG(INFO) << "data reader buffer is empty";
      }
      return result;
    }

    void getBufferState( int& current, int& maximum )
    {
      maximum = max_buffer_size;
      std::lock_guard<std::mutex> lock(buffer_mutex);
      current = buffer.size();
    }

  private:


    Scene_sptr create_scene()
    {
      std::vector<double> source_weights;
      int iter = iteration->load();
      for( size_t i = 0; i < sources.size(); ++i )
        source_weights.push_back(sources[i]->weight(iter));

      std::discrete_distribution<int> source_dist(source_weights.begin(), source_weights.end());
      int source_i = source_dist(rng);
      Scene_sptr scene;
      while( !scene )
        scene = sources[source_i]->create_scene();
      sources[source_i]->advance();
      return scene;
    }


    void run()
    {

      Scene_sptr scene;
      while(!quit.load())
      {

        if(!scene)
        {
          scene = create_scene();
        }

        int current_buffer_size;
        {
          std::lock_guard<std::mutex> lock(buffer_mutex);
          current_buffer_size = buffer.size();
          if(scene && current_buffer_size < max_buffer_size)
          {
            buffer.push(scene);
            scene.reset();
          }
        }

        if( current_buffer_size == max_buffer_size )
        {
          // buffer is full give cpu a break
          //std::this_thread::sleep_for(std::chrono::milliseconds(4)); 
          std::this_thread::yield();
        }
      }
      // cleanup 
      std::lock_guard<std::mutex> lock(buffer_mutex);
      while( buffer.size() )
      {
        buffer.pop();
      }
    }

    const MultiViH5Params param;
    std::atomic<int>* iteration;
    const int max_buffer_size;
    std::queue<Scene_sptr> buffer;

    std::vector<std::unique_ptr<Source>> sources;
    std::mt19937 rng;

    std::atomic<bool> quit;
    std::mutex buffer_mutex; // mutex for accessing the buffer
    std::thread thread;
  };



  template <class Dtype>
  class BatchBuilder
  {
  public:

    struct Batch
    {
      MyBlob<Dtype> images;
      MyBlob<Dtype> depths;
      MyBlob<Dtype> flow;
      MyBlob<Dtype> motion;
      MyBlob<Dtype> intrinsics;
      MyBlob<Dtype> depthmasks;
      std::string sample_ids;
    };
    typedef std::shared_ptr<Batch> Batch_sptr;

    BatchBuilder(DataReader* data_source, const MultiViH5Params& param, TopIndices top_idx)
      :data_source(data_source),
      param(param),
      top_idx(top_idx),
      batch_size(param.batch_size),
      quit(false)
    {
      scene_pool_size = param.scene_pool_size;
      // fill the pool
      pool.resize(scene_pool_size);
      for( auto& item : pool )
      {
        Scene_sptr scene = data_source->getData();
        prepareScene(scene);
        item.first = scene;
        item.second = 0;
      }
    }


    void start()
    {
      int num_threads = 0; 
      // set num_threads using the OMP limit
      char* omp_num_threads = std::getenv("OMP_NUM_THREADS");
      if( omp_num_threads )
      {
        num_threads = std::stoi(omp_num_threads);
      }
      // overwrite with the value given in the prototxt
      if( param.builder_threads.is_valid() || num_threads <= 0 )
      {
        num_threads = param.builder_threads.value();
      }

      // use only 1 thread in test mode to keep the order of data deterministic
      if( param.test_phase )
        num_threads = 1;
#ifdef DEBUG_SINGLE_THREAD
      num_threads = 1;
#endif

      threads.resize(num_threads);
      for( size_t i = 0; i < threads.size(); ++i )
      {
        // small delay because we init the rng with the current time in each
        // thread
        std::this_thread::sleep_for(std::chrono::milliseconds(4)); 
        threads[i] = std::thread(&BatchBuilder::run, this);
      }
    }

    void stop()
    {
      quit = true;
      for( size_t i = 0; i < threads.size(); ++i )
        if( threads[i].joinable() )
          threads[i].join();

      {
        std::lock_guard<std::mutex> lock(batches_mutex);
        while( batches.size() )
        {
          batches.pop();
        }
      }
    }

    Batch_sptr getBatch()
    {
      bool print_warning = !param.test_phase;
      Batch_sptr result = nullptr;
      while( !result )
      {
        std::lock_guard<std::mutex> lock(batches_mutex);
        if( batches.size() )
        {
          result = batches.front();
          batches.pop();
        }
        else
        {
          if(print_warning)
          {
            static int empty_count = 0;
            ++empty_count;
            if( !((empty_count-1)%100) )
              std::cerr << "batch queue is empty (" << empty_count << " warnings)\n";
          }
          print_warning = false;
        }
      }
      return result;
    }

    // call start() before calling this function
    void getShapes(
        std::vector<int>& images_shape,
        std::vector<int>& depths_shape,
        std::vector<int>& flow_shape,
        std::vector<int>& motion_shape,
        std::vector<int>& intrinsics_shape,
        std::vector<int>& depthmasks_shape
        )
    {
      Batch_sptr b;
      while( !b )
      {
        std::lock_guard<std::mutex> lock(batches_mutex);
        //LOG_IF(INFO,batches.size()>1) << "batch builder buffer size " << batches.size();
        if( batches.size() )
        {
          b = batches.front();
          images_shape     = b->images.shape();
          depths_shape     = b->depths.shape();
          flow_shape       = b->flow.shape();
          motion_shape     = b->motion.shape();
          intrinsics_shape = b->intrinsics.shape();
          depthmasks_shape = b->depthmasks.shape();
        }
      }
    }

    void getBufferState( int& current, int& maximum )
    {
      maximum = MAX_PRELOADED_BATCHES;
      std::lock_guard<std::mutex> lock(batches_mutex);
      current = batches.size();
    }

  private:

    
    void prepareScene(Scene_sptr scene)
    {
      const bool scale = param.scaled_width.is_valid() ||
                         param.scaled_height.is_valid();
      for( CameraVec& camvec : scene->cameras )
      {
        for( Camera& cam : camvec )
        {
          // normalize intrinsics
          cam.K(0,0) /= cam.width;
          cam.K(1,1) /= cam.height;
          cam.K(0,2) /= cam.width;
          cam.K(1,2) /= cam.height;

          int scaled_width, scaled_height;
          if( scale )
          {
            scaled_width = param.scaled_width.value();
            scaled_height = param.scaled_height.value();
          }
          else
          {
            scaled_width = cam.width;
            scaled_height = cam.height;
          }


          if( cam.image_data.size() )
          {
            WebPDecoderConfig config;
            WebPInitDecoderConfig(&config);

            WebPGetFeatures((const uint8_t*)cam.image_data.data(), cam.image_data.size(), &config.input);
            cam.width = config.input.width;
            cam.height = config.input.height;
            cam.channels = 3;


            config.options.bypass_filtering = 1;
            config.options.no_fancy_upsampling = 1;
      
            std::string tmp_data;
            tmp_data.resize(cam.width*cam.height*cam.channels);
              
            config.output.colorspace = MODE_BGR;
            config.output.u.RGBA.rgba = (uint8_t*) &tmp_data[0];
            config.output.u.RGBA.stride = cam.channels*cam.width;
            config.output.u.RGBA.size = config.output.u.RGBA.stride * cam.height;
            config.output.is_external_memory = 1;
            WebPDecode((const uint8_t*)cam.image_data.data(), cam.image_data.size(), &config);
            
            if( scale )
            {
              cv::Mat img(cam.height, cam.width, CV_8UC3, &tmp_data[0]);
              cv::Mat scaled_img;
              cv::resize(img, scaled_img, cv::Size(scaled_width, scaled_height),
                         0, 0, cv::INTER_AREA);
              assert(scaled_img.isContinuous());
              cam.image_data.resize(scaled_width*scaled_height*3);
              memcpy(&cam.image_data[0],scaled_img.ptr<char>(), cam.image_data.size());
            }
            else
            {
              cam.image_data.swap(tmp_data);
            }


            if( param.convert_to_gray_values )
            {
              convertBGRToFloatGray<float>(cam.image_data,
                  param.image_range_min,
                  param.image_range_max);
              cam.channels = 1;
            }
            else
            {
              convertBGRToFloatRGBLayers<float>(
                  cam.image_data,scaled_width,scaled_height,
                  param.image_range_min,
                  param.image_range_max);
            }
          }

          if( cam.depth_data.size() )
          {
            std::vector<half_float::half> depth_half(cam.width*cam.height);
            int decompressed_size = depth_half.size() * sizeof(half_float::half);
            LZ4_decompress_safe(&cam.depth_data[0], (char*)&depth_half[0], cam.depth_data.size(), decompressed_size);
            cam.depth_data.resize(depth_half.size()*sizeof(float));
            float* depth_float = (float*)&cam.depth_data[0];
            for( int i = 0; i < (int)depth_half.size(); ++i )
              depth_float[i] = half_float::half_cast<float,std::round_to_nearest>(depth_half[i]);

            if( scale )
            {
              cv::Mat depth(cam.height, cam.width, CV_32FC1, &depth_float[0]);
              cv::Mat scaled_depth;
              cv::resize(depth, scaled_depth, cv::Size(scaled_width, scaled_height),
                  0, 0, cv::INTER_NEAREST);
              assert(scaled_depth.isContinuous());
              cam.depth_data.resize(scaled_height*scaled_width*sizeof(float));
              memcpy(&cam.depth_data[0], scaled_depth.ptr<char>(), cam.depth_data.size());
            }

            // convert to CAMERA_Z
            if( cam.depth_mode == Camera::RAY_LENGTH )
            {
              cam.depth_mode = Camera::CAMERA_Z;
              Eigen::Matrix3f K = cam.K.cast<float>();
              K(0,0) *= scaled_width;
              K(1,1) *= scaled_height;
              K(0,2) *= scaled_width;
              K(1,2) *= scaled_height;
              Eigen::Matrix3f inv_K = K.inverse();
              
              int i = 0;
              for( int y = 0; y < scaled_height; ++y )
              for( int x = 0; x < scaled_width; ++x, ++i )
              {
                Eigen::Vector2f p1(x+0.5f,y+0.5f);
                Eigen::Vector3f pos;
                pos.x() = inv_K(0,0)*p1.x() + inv_K(0,2);
                pos.y() = inv_K(1,1)*p1.y() + inv_K(1,2);
                pos.z() = 1;
                float norm = pos.norm();
                depth_float[i] /= norm;
              }

            }
          }

          // update image dimensions
          cam.width = scaled_width;
          cam.height = scaled_height;
        } // cam
      } // camvec
    }

    void run()
    {
      Scene_sptr prepared_scene;
      Batch_sptr prepared_batch;

      std::uniform_int_distribution<int> dist_scene(0,pool.size()-1);
      std::bernoulli_distribution dist_rot180(param.augment_rot180);
      std::bernoulli_distribution dist_mirrorX(param.augment_mirror_x);
      std::mt19937 rng;
      rng.seed(std::chrono::high_resolution_clock::now().time_since_epoch().count());
      int scene_idx = -1;

      while( !quit )
      {
        if( !prepared_scene )
        {
          prepared_scene = data_source->getData();
          prepareScene(prepared_scene);
        }

        // build a batch
        if( !prepared_batch )
        {
          prepared_batch.reset(new Batch());
          int batch_idx = 0; // the item index within the batch
          while( batch_idx < batch_size )
          {
            if( !prepared_scene )
            {
              prepared_scene= data_source->getData();
              prepareScene(prepared_scene);
            }

            if( param.test_phase )
            {
              // deterministic scene_idx
              scene_idx = (scene_idx+1) % pool.size();
            }
            else
            {
              // randomly select a scene from the pool
              scene_idx = dist_scene(rng);
            }

            std::shared_ptr<Scene> selected_scene;
            std::pair<int,int> selected_image_pair;
            {
              std::lock_guard<std::mutex> lock(pool_mutex);

              selected_scene = pool[scene_idx].first;
              selected_image_pair = selected_scene->combinations[pool[scene_idx].second];
              pool[scene_idx].second += 1;

              // if all combinations have been used replace the scene in the pool
              // with the preloaded prepared scene
              if( pool[scene_idx].second == (int)selected_scene->combinations.size() )
              {
                pool[scene_idx].first = prepared_scene;
                pool[scene_idx].second = 0;
                prepared_scene.reset();
              }
            }

            // decide whether to rotate and mirror or not
            bool rotate180, mirrorX;
            if( !param.test_phase )
            {
               rotate180 = dist_rot180(rng);
               mirrorX = dist_mirrorX(rng);
            }
            else
            {
              rotate180 = batch_idx < dist_rot180.p()*batch_size;
              mirrorX = batch_idx < dist_mirrorX.p()*batch_size;
            }

            CameraVec& camvec1 = selected_scene->cameras[selected_image_pair.first];
            CameraVec& camvec2 = selected_scene->cameras[selected_image_pair.second];
            for( size_t subcam_idx = 0; subcam_idx < camvec1.size() && batch_idx < batch_size; ++subcam_idx)
            {
              Camera& cam1 = camvec1[subcam_idx];
              Camera& cam2 = camvec2[subcam_idx];

              //debug
              {
                //cv::Mat img0(images_shape[2],images_shape[3], cv::DataType<Dtype>::type, &cam1.image_data[0]);
                //cv::Mat img1(images_shape[2],images_shape[3], cv::DataType<Dtype>::type, &cam2.image_data[0]);
                //cv::imshow("img0", img0);
                //cv::imshow("img1", img1);
                //cv::waitKey(0);
              }

              if( top_idx.images >= 0 )
              {
                // copy first image
                std::vector<int> shape{
                  batch_size, 
                  cam1.channels+cam2.channels, 
                  cam1.height, 
                  cam1.width };
                if( shape != prepared_batch->images.shape() )
                  prepared_batch->images.Reshape(shape);

                int offset = prepared_batch->images.offset(batch_idx,0);
                Dtype* dst = prepared_batch->images.mutable_cpu_data()+offset;
                Dtype* src = (Dtype*)&cam1.image_data[0];
                memcpy(dst, src, cam1.image_data.size());

                // copy second image
                offset = prepared_batch->images.offset(batch_idx,cam1.channels);
                dst = prepared_batch->images.mutable_cpu_data()+offset;
                src = (Dtype*)&cam2.image_data[0];
                memcpy(dst, src, cam2.image_data.size());

                if( rotate180 )
                  rotateImageBy180(prepared_batch->images, batch_idx);
                if( mirrorX )
                  mirrorImageX(prepared_batch->images, batch_idx);

                // augment image values
                augmentImage(
                    prepared_batch->images, 
                    batch_idx, 
                    param.source[selected_scene->source_id], 
                    rng);
              }
              

              double depth_scale_factor = 1;
              // generate and copy the motion
              {
                Eigen::Matrix3d R1 = cam1.R;
                Eigen::Matrix3d R2 = cam2.R;
                Eigen::Vector3d t1 = cam1.t;
                Eigen::Vector3d t2 = cam2.t;
                if( rotate180 )
                {
                  rotateCamera180DegAroundZ(R1,t1);
                  rotateCamera180DegAroundZ(R2,t2);
                }

                // rotation and translation from cam1 to cam2.
                // R12,t12 transform a point X1 in the coordinate frame of cam1 to
                // the coordinate frame of cam2
                Eigen::Matrix3d R12 = R2*R1.transpose();
                Eigen::Vector3d t12 = t2-R12*t1;

                if( mirrorX )
                {
                  Eigen::Vector3d C2 = -R12.transpose()*t12;
                  C2.x() = -C2.x();
                  R12.col(0) *= -1;
                  R12.row(0) *= -1;
                  t12 = -R12*C2;
                }

                // a single camera can only predict the direction of the movement 
                // -> normalize translation
                if( t12.norm() < 1e-6 )
                  continue; // skip image pair
                // normalize the translation and scale the depth accordingly
                if (param.norm_trans_scale_depth)
                {
                  depth_scale_factor = 1/t12.norm();
                  t12.normalize(); 
                }

                switch( param.motion_format )
                {
                  case MultiViH5Params::MotionFormat::ANGLEAXIS6:
                  {
                    std::vector<int> shape{batch_size, 6};
                    if( shape != prepared_batch->motion.shape() )
                      prepared_batch->motion.Reshape(shape);
                    double aa_data[3];
                    rotationMatrixToAngleAxis(aa_data, R12);

                    int offset = prepared_batch->motion.offset(batch_idx);
                    Dtype* dst = prepared_batch->motion.mutable_cpu_data()+offset;
                    dst[0] = aa_data[0];
                    dst[1] = aa_data[1];
                    dst[2] = aa_data[2];
                    dst[3] = t12.x();
                    dst[4] = t12.y();
                    dst[5] = t12.z();
                  }
                  break;
                case MultiViH5Params::MotionFormat::ANGLEAXIS7:
                  {
                    std::vector<int> shape{batch_size, 7};
                    if( shape != prepared_batch->motion.shape() )
                      prepared_batch->motion.Reshape(shape);
                    double aa_data[3];
                    Eigen::Map<Eigen::Vector3d> aa(aa_data);
                    rotationMatrixToAngleAxis(aa_data, R12);
                    double magnitude = aa.norm();
                    if( magnitude < 1e-6 )
                      aa.setZero();
                    else
                      aa.normalize();

                    int offset = prepared_batch->motion.offset(batch_idx);
                    Dtype* dst = prepared_batch->motion.mutable_cpu_data()+offset;
                    dst[0] = magnitude;
                    dst[1] = aa.x();
                    dst[2] = aa.y();
                    dst[3] = aa.z();
                    dst[4] = t12.x();
                    dst[5] = t12.y();
                    dst[6] = t12.z();
                  }
                  break;
                case MultiViH5Params::MotionFormat::QUATERNION:
                  {
                    std::vector<int> shape{batch_size, 7};
                    if( shape != prepared_batch->motion.shape() )
                      prepared_batch->motion.Reshape(shape);
                    Eigen::Quaterniond q(R12);
                    int offset = prepared_batch->motion.offset(batch_idx);
                    Dtype* dst = prepared_batch->motion.mutable_cpu_data()+offset;
                    dst[0] = q.w();
                    dst[1] = q.x();
                    dst[2] = q.y();
                    dst[3] = q.z();
                    dst[4] = t12.x();
                    dst[5] = t12.y();
                    dst[6] = t12.z();
                  }
                  break;
                case MultiViH5Params::MotionFormat::FMATRIX:
                  {
                    std::vector<int> shape{batch_size, 8};
                    if( shape != prepared_batch->motion.shape() )
                      prepared_batch->motion.Reshape(shape);
                    Eigen::Matrix<double,3,4> P1, P2;
                    P1 << R1, t1;
                    P1 = cam1.K * P1;
                    P2 << R2, t2;
                    P2 = cam2.K * P2;
                    Eigen::Matrix3d F = computeFundamentalFromCameras(P1,P2);
                    double normalizer = 1/F(2,2); // make everything relative to the last element


                    if(std::abs(F(2,2)) < 1e-6)
                    {
                      std::cerr << "skipping batch item. F(2,2) is too small\n";
                      continue;
                    }

                    // copy the normalized fmatrix
                    int offset = prepared_batch->motion.offset(batch_idx);
                    Dtype* dst = prepared_batch->motion.mutable_cpu_data()+offset;
                    int linear_idx = 0;
                    for(int j = 0; j < 3; ++j)
                    for(int i = 0; i < 3 && linear_idx < 8; ++i, ++linear_idx)
                      dst[linear_idx] = F(i,j)*normalizer;
                  }
                  break;
                }// switch motion_format
              }

              // intrinsics
              if( top_idx.intrinsics >= 0 )
              {
                std::vector<int> shape{batch_size, 4};
                if( shape != prepared_batch->intrinsics.shape() )
                  prepared_batch->intrinsics.Reshape(shape);

                Dtype fx = cam1.K(0,0);
                Dtype fy = cam1.K(1,1);

                Dtype cx = cam1.K(0,2);
                Dtype cy = cam1.K(1,2);

                if( rotate180 )
                {
                  // rotate the principal point
                  cx = 1-cx;
                  cy = 1-cy;
                }
                if( mirrorX )
                {
                  cx = 1-cx; // mirror the x component
                }

                // copy the normalized intrinsics
                int offset = prepared_batch->intrinsics.offset(batch_idx);
                Dtype* dst = prepared_batch->intrinsics.mutable_cpu_data()+offset;
                dst[0] = fx;
                dst[1] = fy;
                dst[2] = cx;
                dst[3] = cy;
              }

              // copy flow
              if( top_idx.flow >= 0 )
              {
                std::vector<int> shape{batch_size, 2, cam1.height, cam1.width};
                if( shape != prepared_batch->flow.shape() )
                  prepared_batch->flow.Reshape(shape);
                int offset = prepared_batch->flow.offset(batch_idx);
                Dtype* dst = prepared_batch->flow.mutable_cpu_data()+offset;
                computeFlow(dst, cam1, cam2);

                if( rotate180 )
                {
                  rotateImageBy180(prepared_batch->flow, batch_idx);
                  // negate the flow vectors
                  dst = prepared_batch->flow.mutable_cpu_data();
                  int offset2 = prepared_batch->flow.offset(batch_idx+1);
                  for( Dtype* it = dst+offset; it < dst+offset2; ++it )
                    *it = -(*it);
                }
                if( mirrorX )
                {
                  mirrorImageX(prepared_batch->flow, batch_idx);
                  // negate the x component of the flow vectors
                  dst = prepared_batch->flow.mutable_cpu_data();
                  int offset2 = prepared_batch->flow.offset(batch_idx, 1);
                  for( Dtype* it = dst+offset; it < dst+offset2; ++it )
                    *it = -(*it);
                }
              }

              // copy depth
              if( top_idx.depths >= 0 )
              {	
                std::vector<int> shape{
                  batch_size, 
                    param.depth_pair ? 2 : 1, 
                    cam1.height, 
                    cam1.width};
                if( shape != prepared_batch->depths.shape() )
                  prepared_batch->depths.Reshape(shape);

                Dtype max_data = param.max_depth;
                Dtype min_data = param.min_depth;
                bool inv_depth = param.inverse_depth;

                // copy the first depth
                {
                  int offset = prepared_batch->depths.offset(batch_idx,0);
                  Dtype* dst = prepared_batch->depths.mutable_cpu_data()+offset;
                  Dtype* src = (Dtype*)&cam1.depth_data[0];
                  memcpy(dst, src, cam1.depth_data.size());
                }

                // copy the second depth
                if( param.depth_pair )
                {
                  int offset = prepared_batch->depths.offset(batch_idx,1);
                  Dtype* dst = prepared_batch->depths.mutable_cpu_data()+offset;
                  Dtype* src = (Dtype*)&cam2.depth_data[0];
                  memcpy(dst, src, cam2.depth_data.size());
                }
                {
                  int offset = prepared_batch->depths.offset(batch_idx,0);
                  Dtype* dst = prepared_batch->depths.mutable_cpu_data()+offset;
                  
                  for(int i = 0; i < prepared_batch->depths.count(1); ++i )
                  {
                    if( dst[i] == 0 ) // zero depth is always invalid
                    {
                      dst[i] = NAN;
                      continue;
                    }
                    if(max_data > 0 && dst[i] > max_data)
                    {
                      dst[i] = NAN;
                      continue;
                    }
                    if(min_data > 0 && dst[i] < min_data)
                    {
                      dst[i] = NAN;
                      continue;
                    }
                    
                    dst[i] *= depth_scale_factor;
                    if(inv_depth)
                      dst[i] = 1/dst[i];
                  }
                }
        
                if( rotate180 )
                  rotateImageBy180(prepared_batch->depths, batch_idx);
                if( mirrorX )
                  mirrorImageX(prepared_batch->depths, batch_idx );
              }

              if( top_idx.depthmasks >= 0 )
              {
                const int border1 = param.depthmask_border1;
                const int border2 = param.depthmask_border2;

                std::vector<int> shape{
                  batch_size, 
                    param.depth_pair ? 2 : 1, 
                    cam1.height, 
                    cam1.width};
                if( shape != prepared_batch->depthmasks.shape() )
                  prepared_batch->depthmasks.Reshape(shape);

                // copmute the first depthmask
                int offset = prepared_batch->depthmasks.offset(batch_idx,0);
                Dtype* dst = prepared_batch->depthmasks.mutable_cpu_data()+offset;
                computeDepthmask(dst, cam1, cam2, border1, border2);

                // compute the second depthmask
                if( param.depth_pair )
                {
                  offset = prepared_batch->depthmasks.offset(batch_idx,1);
                  dst = prepared_batch->depthmasks.mutable_cpu_data()+offset;
                  computeDepthmask(dst, cam2, cam1, border1, border2);
                }
                if( rotate180 )
                  rotateImageBy180(prepared_batch->depthmasks, batch_idx);
                if( mirrorX )
                  mirrorImageX(prepared_batch->depthmasks, batch_idx);

              }

              // sample ids
              {
                std::stringstream sample_id;
                sample_id << selected_scene->scene_id << "/frames/t0/v" << selected_image_pair.first << ",v" << selected_image_pair.second << "\n";
                prepared_batch->sample_ids += sample_id.str();
              }

              ++batch_idx;
            }
          }

        } 
        {
          batches_mutex.lock();
          if( prepared_batch && batches.size() < MAX_PRELOADED_BATCHES )
          {
            batches.push(prepared_batch);
            batches_mutex.unlock();
            prepared_batch.reset();
          }
          else
          {
            batches_mutex.unlock();
            // buffer is full give cpu a break
            std::this_thread::sleep_for(std::chrono::milliseconds(4)); 
            //std::this_thread::yield();
          }
        }
      } // while

    } // run()

    int scene_pool_size;
    DataReader* data_source;
    const MultiViH5Params param;
    TopIndices top_idx;
    const int batch_size;

    // pool of viewpointsets.
    // .second is the number of already used combinations
    std::vector<std::pair<Scene_sptr,int>> pool;
    std::mutex pool_mutex;  // mutex for read_queue


    std::queue<Batch_sptr> batches;
    std::mutex batches_mutex;  // mutex for read_queue

    std::atomic<bool> quit;

    std::vector<std::thread> threads;
  };


}
using namespace multiviewh5datareader_internal;


struct MultiViH5DataReader::PrivateData
{
  PrivateData(MultiViH5DataReader* parent, const MultiViH5Params& params )
    :parent(parent), params(params), test_iterations(0), iteration(0)
  { 
  }
  ~PrivateData()
  {
  }

  void init()
  {
    // identify the required top blobs
    if( params.top_output.count(MultiViH5Params::IMAGE_PAIR) )
      top_idx.images = 0;
    if( params.top_output.count(MultiViH5Params::MOTION) )
      top_idx.motion = 1;
    if( params.top_output.count(MultiViH5Params::DEPTH) )
      top_idx.depths = 2;
    if( params.top_output.count(MultiViH5Params::FLOW) )
      top_idx.flow = 3;
    if( params.top_output.count(MultiViH5Params::INTRINSICS) )
      top_idx.intrinsics = 4;
    if( params.top_output.count(MultiViH5Params::DEPTHMASKS) )
      top_idx.depthmasks = 5;

    if( params.test_phase )
    {
      const int num_sources = params.source.size();
        
      if( num_sources != 1 )
        throw std::runtime_error("The number of sources must be exactly 1 in test phase! ");

      std::vector<std::string> h5_paths = split(
          params.source[0].path,
          ';');
      int num_samples = countSampleGroups(h5_paths);
      int batch_size = params.batch_size;

      div_t tmp = div(num_samples, batch_size);
      if( tmp.rem == 0 )
      {
        std::cerr << "number of test samples = " << num_samples 
          << ",  test_iter = " << tmp.quot << std::endl;
      }
      else
      {
        std::stringstream errstr;
        errstr << "test_iter cannot be set to process all samples exactly one\n"
          << "num_samples/batch_size = " << num_samples << "/" << batch_size << " = "
          << tmp.quot << " rem " << tmp.rem << std::endl;
        throw std::runtime_error(errstr.str());
      }
      test_iterations = tmp.quot;
    }

    // start data reader
    data_reader.reset(new DataReader(params, top_idx, &iteration));
    data_reader->start();

    // start batch builder
    batch_builder.reset(new BatchBuilder<float>(data_reader.get(), params, top_idx));
    batch_builder->start();
    batch_builder->getShapes(
      images_shape,
      depths_shape,
      flow_shape,
      motion_shape,
      intrinsics_shape,
      depthmasks_shape
    );

  }

  MultiViH5DataReader* parent; // the MultiViH5DataLayer 
  const MultiViH5Params params;
  int test_iterations;
  std::atomic<int> iteration;

  std::unique_ptr<DataReader> data_reader;
  std::unique_ptr<BatchBuilder<float>> batch_builder;
      
  std::vector<int> images_shape;
  std::vector<int> depths_shape;
  std::vector<int> flow_shape;
  std::vector<int> motion_shape;
  std::vector<int> intrinsics_shape;
  std::vector<int> depthmasks_shape;

  TopIndices top_idx;

};


MultiViH5DataReader::MultiViH5DataReader( const MultiViH5Params* _params)
{
  //mtrace();
  MultiViH5Params params = *_params;
  if( params.scaled_width.is_valid() != params.scaled_height.is_valid() )
    throw std::runtime_error("Both dimensions must be specified for scaling");
  if( params.top_output.empty() )
    throw std::runtime_error("At least one output must be specified");
  d = new PrivateData(this, params );
  d->init();
}

MultiViH5DataReader::~MultiViH5DataReader() 
{
  if( d->batch_builder )
    d->batch_builder->stop();
  if( d->data_reader )
    d->data_reader->stop();
  delete d;
  //muntrace();
}

void MultiViH5DataReader::getShape( 
      std::vector<int>& imagepair, 
      std::vector<int>& motion,
      std::vector<int>& flow,
      std::vector<int>& depth,
      std::vector<int>& intrinsics,
      std::vector<int>& depthmasks )
{
  d->batch_builder->getShapes(
    d->images_shape,
    d->depths_shape,
    d->flow_shape,
    d->motion_shape,
    d->intrinsics_shape,
    d->depthmasks_shape
  );

  imagepair = d->images_shape;
  motion = d->motion_shape;
  flow = d->flow_shape;
  depth = d->depths_shape;
  intrinsics = d->intrinsics_shape;
  depthmasks = d->depthmasks_shape;
}

int MultiViH5DataReader::getData( 
      float* imagepair, 
      float* motion,
      float* flow,
      float* depth,
      float* intrinsics,
      float* depthmasks,
      std::string* sample_ids )
{
  typename BatchBuilder<float>::Batch_sptr batch = d->batch_builder->getBatch();

  if( imagepair && d->top_idx.images >= 0 )
  {
    memcpy(imagepair, batch->images.cpu_data(), batch->images.byte_size());
  }
  if( motion && d->top_idx.motion >= 0 )
  {
    memcpy(motion, batch->motion.cpu_data(), batch->motion.byte_size());
  }
  if( flow && d->top_idx.flow >= 0 )
  {
    memcpy(flow, batch->flow.cpu_data(), batch->flow.byte_size());
  }
  if( depth && d->top_idx.depths >= 0 )
  {
    memcpy(depth, batch->depths.cpu_data(), batch->depths.byte_size());
  }
  if( intrinsics && d->top_idx.intrinsics >= 0 )
  {
    memcpy(intrinsics, batch->intrinsics.cpu_data(), batch->intrinsics.byte_size());
  }
  if( depthmasks && d->top_idx.depthmasks >= 0 )
  {
    memcpy(depthmasks, batch->depthmasks.cpu_data(), batch->depthmasks.byte_size());
  }
  if( sample_ids )
  {
    sample_ids->swap( batch->sample_ids );
  }

  return d->params.batch_size;
}

void MultiViH5DataReader::setIteration( int iteration )
{
  d->iteration = iteration;
}


size_t MultiViH5DataReader::getNumberOfTestIterations() const
{
  return d->test_iterations;
}


void MultiViH5DataReader::getBufferStates( std::pair<int,int>& batch_buffer, std::pair<int,int>& read_buffer )
{
  d->batch_builder->getBufferState( batch_buffer.first, batch_buffer.second );
  d->data_reader->getBufferState( read_buffer.first, read_buffer.second );
}

