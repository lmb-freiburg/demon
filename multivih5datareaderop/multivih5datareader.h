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
#ifndef MULTIVIH5READER_H_
#define MULTIVIH5READER_H_
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <string>
#include <set>
#include "json.hpp"

template<class T>
class Optional
{
public:
  Optional()
    :valid_(false)
  { }

  Optional(const T& value, bool valid)
    :value_(value), valid_(valid)
  { }

  Optional<T>& operator=(const T& other)
  {
    value_ = other;
    valid_ = true;
    return *this;
  }

  T* operator->() 
  {
    return &value_;
  }
  
  const T* operator->() const
  {
    return &value_;
  }

  T& operator*() 
  {
    return value_;
  }

  const T& operator*() const
  {
    return value_;
  }

  operator bool() const
  {
    return valid_;
  }

  bool is_valid() const
  {
    return valid_;
  }

  T& value()
  {
    return value_;
  }

  const T& value() const
  {
    return value_;
  }

//private:
  T value_;
  bool valid_;
};

class MultiViH5Params
{
public:
  enum MotionFormat
  {
        FMATRIX = 0,    // normalized fundamental matrix
                        // translation is always a 3d vector 
        ANGLEAXIS6 = 1, // angle axis with varying magnitude
        ANGLEAXIS7 = 2, // normalized angle axis with rotation angle stored as additional parameter
        QUATERNION = 3
  };

  enum Output
  {
        IMAGE_PAIR = 0,
        MOTION = 1,
        FLOW = 2,
        DEPTH = 3,
        INTRINSICS = 4,
        DEPTHMASKS = 5
  };

  MultiViH5Params()
    :batch_size(1),
    test_phase(false),
    augment_rot180(0.5),
    augment_mirror_x(0.5),
    image_range_min(-0.5),
    image_range_max(0.5),
    convert_to_gray_values(false),
    motion_format(FMATRIX),
    depth_pair(false),
    norm_trans_scale_depth(true),
    inverse_depth(false),
    min_depth(-1),
    max_depth(-1),
    builder_threads(4,false),
    scaled_width(0,false),
    scaled_height(0,false),
    scene_pool_size(64),
    depthmask_border1(3),
    depthmask_border2(5)
  { }

  int batch_size;
  bool test_phase;

  // The probability for rotating images by 180 deg.
  float augment_rot180;
  // The probability to mirror the images along the x axis
  float augment_mirror_x;

  // maps [0,255] to [image_range_min, image_range_max]
  float image_range_min;
  float image_range_max;
  // converts the images to gray values
  bool convert_to_gray_values;
  
  MotionFormat motion_format;
  bool depth_pair;
  bool norm_trans_scale_depth;
  bool inverse_depth;
  float min_depth;
  float max_depth;
  Optional<uint32_t> builder_threads;

  Optional<int32_t> scaled_width;
  Optional<int32_t> scaled_height;

  int32_t scene_pool_size;


  // All pixel correspondences must be within the images.
  // Additionally, the pixels must not lie in the border regions.
  // border1 corresponds to the first image
  // border2 corresponds to the second image
  int32_t depthmask_border1;
  int32_t depthmask_border2;

  // Use top_output if you want to give arbirary names to the tops
  // If used the number of top_output must match the number of tops.
  std::set<Output> top_output;

  class Source
  {
  public:
    struct KeyFrame 
    {
      enum Interpolation { LINEAR = 0 };
      KeyFrame()
        :t(0),v(1),i(LINEAR)
      { }

      uint32_t t;
      float v;
      Interpolation i;
    };

    class RandomParam 
    {
    public:

      struct Normal 
      {
        Normal()
          :mean(0),stddev(1)
        { }
        float mean;
        float stddev;
      };
      struct Uniform 
      {
        Uniform()
          :a(0),b(1)
        { }
        float a;
        float b;
      };

      // choose one
      Optional<Normal> normal;
      Optional<Uniform> uniform;
    };

    std::string path; // list of hdf5 files separated with a semicolon
    // The weight defines how often data is read from this source.
    // The weight can be changed over time using multiple keyframes
    std::vector<KeyFrame> weight;

    // augmentations in hsv color space will be applied first
    Optional<RandomParam> aug_hsv_hue;
    Optional<RandomParam> aug_hsv_sat;
    Optional<RandomParam> aug_hsv_val;
    Optional<RandomParam> aug_contrast;
    Optional<RandomParam> aug_brightness;
    Optional<RandomParam> aug_gamma;


  }; // Source
  std::vector<Source> source;
};


class MultiViH5DataReader
{
public:
  MultiViH5DataReader( const MultiViH5Params* params );
  MultiViH5DataReader( const MultiViH5DataReader& ) = delete;

  ~MultiViH5DataReader();
  

  void getShape( 
      std::vector<int>& imagepair, 
      std::vector<int>& motion,
      std::vector<int>& flow,
      std::vector<int>& depth,
      std::vector<int>& intrinsics,
      std::vector<int>& depthmasks );

  int getData( 
      float* imagepair=0, 
      float* motion=0,
      float* flow=0,
      float* depth=0,
      float* intrinsics=0,
      float* depthmasks=0,
      std::string* sample_ids=0 );

  void setIteration( int iteration );

  size_t getNumberOfTestIterations() const;

  void getBufferStates( std::pair<int,int>& batch_buffer, std::pair<int,int>& read_buffer );

private:
  struct PrivateData;
  PrivateData* d;

};

// Returns 0 on success
inline int MultiViH5Params_from_json(const std::string& json_str, MultiViH5Params* params)
{
  using json = nlohmann::json;
  auto root = json::parse(json_str);

  for( auto it = root.begin(); it != root.end(); ++it )
  {
    if( it.key() == "batch_size" )
    {
      params->batch_size = it.value().get<int>();
    }
    else if( it.key() == "test_phase" )
    {
      params->test_phase = it.value().get<bool>();
    } 
    else if( it.key() == "augment_rot180" )
    {
      params->augment_rot180 = it.value().get<double>();
    } 
    else if( it.key() == "augment_mirror_x" )
    {
      params->augment_mirror_x = it.value().get<double>();
    } 
    else if( it.key() == "image_range_min" )
    {
      params->image_range_min = it.value().get<double>();
    } 
    else if( it.key() == "image_range_max" )
    {
      params->image_range_max = it.value().get<double>();
    } 
    else if( it.key() == "convert_to_gray_values" )
    {
      params->convert_to_gray_values = it.value().get<bool>();
    } 
    else if( it.key() == "motion_format" )
    {
      std::string motion_format_str = it.value().get<std::string>();
      if( motion_format_str == "FMATRIX" )
        params->motion_format = MultiViH5Params::FMATRIX;
      else if( motion_format_str == "ANGLEAXIS7" )
        params->motion_format = MultiViH5Params::ANGLEAXIS7;
      else if( motion_format_str == "ANGLEAXIS6" )
        params->motion_format = MultiViH5Params::ANGLEAXIS6;
      else if( motion_format_str == "QUATERNION" )
        params->motion_format = MultiViH5Params::QUATERNION;
      else
      {
        std::cerr << "unknown motion format '" << motion_format_str << "'\n";
        return -2;
      }
    } 
    else if( it.key() == "depth_pair" )
    {
      params->depth_pair = it.value().get<bool>();
    } 
    else if( it.key() == "norm_trans_scale_depth" )
    {
      params->norm_trans_scale_depth = it.value().get<bool>();
    } 
    else if( it.key() == "inverse_depth" )
    {
      params->inverse_depth = it.value().get<bool>();
    } 
    else if( it.key() == "min_depth" )
    {
      params->min_depth = it.value().get<double>();
    } 
    else if( it.key() == "max_depth" )
    {
      params->max_depth = it.value().get<double>();
    } 
    else if( it.key() == "builder_threads" )
    {
      params->builder_threads = it.value().get<int>();
    } 
    else if( it.key() == "scaled_width" )
    {
      params->scaled_width = it.value().get<int>();
    } 
    else if( it.key() == "scaled_height" )
    {
      params->scaled_height = it.value().get<int>();
    } 
    else if( it.key() == "scene_pool_size" )
    {
      params->scene_pool_size = it.value().get<int>();
    } 
    else if( it.key() == "depthmask_border1" )
    {
      params->depthmask_border1 = it.value().get<int>();
    } 
    else if( it.key() == "depthmask_border2" )
    {
      params->depthmask_border2 = it.value().get<int>();
    } 
    else if( it.key() == "top_output" )
    {
      auto array = it.value();
      for( auto it = array.begin(); it != array.end(); ++it )
      {
        std::string top_str = it->get<std::string>();
        if( top_str == "IMAGE_PAIR" )
          params->top_output.insert(MultiViH5Params::IMAGE_PAIR);
        else if( top_str == "MOTION" )
          params->top_output.insert(MultiViH5Params::MOTION);
        else if( top_str == "FLOW" )
          params->top_output.insert(MultiViH5Params::FLOW);  
        else if( top_str == "DEPTH" )
          params->top_output.insert(MultiViH5Params::DEPTH);  
        else if( top_str == "INTRINSICS" )
          params->top_output.insert(MultiViH5Params::INTRINSICS);  
        else if( top_str == "DEPTHMASKS" )
          params->top_output.insert(MultiViH5Params::DEPTHMASKS);  
        else
        {
          std::cerr << "unknown top_output '" << top_str << "'\n";
          return -3;
        }
      }
    } 
    else if( it.key() == "source" )
    {
      auto source_array = it.value();
      for( auto source_it = source_array.begin(); source_it != source_array.end(); ++source_it )
      {
        MultiViH5Params::Source source_result;
        if( source_it->count("path") )
          source_result.path = (*source_it)["path"].get<std::string>();

        if( source_it->count("weight") )
        {
          auto weight_array = (*source_it)["weight"];
          for( auto weight_it = weight_array.begin(); weight_it != weight_array.end(); ++weight_it )
          {
            MultiViH5Params::Source::KeyFrame kf;
            if( weight_it->count("t") )
              kf.t = (*weight_it)["t"].get<int>();
            if( weight_it->count("v") )
              kf.v = (*weight_it)["v"].get<double>();
            if( weight_it->count("i") )
            {
              std::string i_str = (*weight_it)["i"].get<std::string>();
              if( i_str == "LINEAR" )
                kf.i = MultiViH5Params::Source::KeyFrame::LINEAR;
              else
              {
                std::cerr << "unknown interpolation type '" << i_str << "'\n";
                return -4;
              }
            }
            source_result.weight.push_back(kf);
          }
        }

        if( source_it->count("aug_hsv_hue") )
        {
          auto aug = (*source_it)["aug_hsv_hue"];
          MultiViH5Params::Source::RandomParam rand_param;
          if( aug.count("normal") )
          {
            auto normal = aug["normal"];
            MultiViH5Params::Source::RandomParam::Normal normal_param;
            if( normal.count("mean") )
              normal_param.mean = normal["mean"].get<double>();
            if( normal.count("stddev") )
              normal_param.stddev = normal["stddev"].get<double>();
            rand_param.normal = normal_param;
          }
          if( aug.count("uniform") )
          {
            auto uniform = aug["uniform"];
            MultiViH5Params::Source::RandomParam::Uniform uniform_param;
            if( uniform.count("a") )
              uniform_param.a = uniform["a"].get<double>();
            if( uniform.count("b") )
              uniform_param.b = uniform["b"].get<double>();
            rand_param.uniform = uniform_param;
          }
          source_result.aug_hsv_hue = rand_param;
        }

        if( source_it->count("aug_hsv_sat") )
        {
          auto aug = (*source_it)["aug_hsv_sat"];
          MultiViH5Params::Source::RandomParam rand_param;
          if( aug.count("normal") )
          {
            auto normal = aug["normal"];
            MultiViH5Params::Source::RandomParam::Normal normal_param;
            if( normal.count("mean") )
              normal_param.mean = normal["mean"].get<double>();
            if( normal.count("stddev") )
              normal_param.stddev = normal["stddev"].get<double>();
            rand_param.normal = normal_param;
          }
          if( aug.count("uniform") )
          {
            auto uniform = aug["uniform"];
            MultiViH5Params::Source::RandomParam::Uniform uniform_param;
            if( uniform.count("a") )
              uniform_param.a = uniform["a"].get<double>();
            if( uniform.count("b") )
              uniform_param.b = uniform["b"].get<double>();
            rand_param.uniform = uniform_param;
          }
          source_result.aug_hsv_sat = rand_param;
        }

        if( source_it->count("aug_hsv_val") )
        {
          auto aug = (*source_it)["aug_hsv_val"];
          MultiViH5Params::Source::RandomParam rand_param;
          if( aug.count("normal") )
          {
            auto normal = aug["normal"];
            MultiViH5Params::Source::RandomParam::Normal normal_param;
            if( normal.count("mean") )
              normal_param.mean = normal["mean"].get<double>();
            if( normal.count("stddev") )
              normal_param.stddev = normal["stddev"].get<double>();
            rand_param.normal = normal_param;
          }
          if( aug.count("uniform") )
          {
            auto uniform = aug["uniform"];
            MultiViH5Params::Source::RandomParam::Uniform uniform_param;
            if( uniform.count("a") )
              uniform_param.a = uniform["a"].get<double>();
            if( uniform.count("b") )
              uniform_param.b = uniform["b"].get<double>();
            rand_param.uniform = uniform_param;
          }
          source_result.aug_hsv_val = rand_param;
        }

        if( source_it->count("aug_contrast") )
        {
          auto aug = (*source_it)["aug_contrast"];
          MultiViH5Params::Source::RandomParam rand_param;
          if( aug.count("normal") )
          {
            auto normal = aug["normal"];
            MultiViH5Params::Source::RandomParam::Normal normal_param;
            if( normal.count("mean") )
              normal_param.mean = normal["mean"].get<double>();
            if( normal.count("stddev") )
              normal_param.stddev = normal["stddev"].get<double>();
            rand_param.normal = normal_param;
          }
          if( aug.count("uniform") )
          {
            auto uniform = aug["uniform"];
            MultiViH5Params::Source::RandomParam::Uniform uniform_param;
            if( uniform.count("a") )
              uniform_param.a = uniform["a"].get<double>();
            if( uniform.count("b") )
              uniform_param.b = uniform["b"].get<double>();
            rand_param.uniform = uniform_param;
          }
          source_result.aug_contrast = rand_param;
        }

        if( source_it->count("aug_brightness") )
        {
          auto aug = (*source_it)["aug_brightness"];
          MultiViH5Params::Source::RandomParam rand_param;
          if( aug.count("normal") )
          {
            auto normal = aug["normal"];
            MultiViH5Params::Source::RandomParam::Normal normal_param;
            if( normal.count("mean") )
              normal_param.mean = normal["mean"].get<double>();
            if( normal.count("stddev") )
              normal_param.stddev = normal["stddev"].get<double>();
            rand_param.normal = normal_param;
          }
          if( aug.count("uniform") )
          {
            auto uniform = aug["uniform"];
            MultiViH5Params::Source::RandomParam::Uniform uniform_param;
            if( uniform.count("a") )
              uniform_param.a = uniform["a"].get<double>();
            if( uniform.count("b") )
              uniform_param.b = uniform["b"].get<double>();
            rand_param.uniform = uniform_param;
          }
          source_result.aug_brightness = rand_param;
        }

        if( source_it->count("aug_gamma") )
        {
          auto aug = (*source_it)["aug_gamma"];
          MultiViH5Params::Source::RandomParam rand_param;
          if( aug.count("normal") )
          {
            auto normal = aug["normal"];
            MultiViH5Params::Source::RandomParam::Normal normal_param;
            if( normal.count("mean") )
              normal_param.mean = normal["mean"].get<double>();
            if( normal.count("stddev") )
              normal_param.stddev = normal["stddev"].get<double>();
            rand_param.normal = normal_param;
          }
          if( aug.count("uniform") )
          {
            auto uniform = aug["uniform"];
            MultiViH5Params::Source::RandomParam::Uniform uniform_param;
            if( uniform.count("a") )
              uniform_param.a = uniform["a"].get<double>();
            if( uniform.count("b") )
              uniform_param.b = uniform["b"].get<double>();
            rand_param.uniform = uniform_param;
          }
          source_result.aug_gamma = rand_param;
        }
         
        params->source.push_back(source_result);
      }
    }
    else
    {
      std::cerr << "unknown key '" << it.key() << "' with value '" << it.value() << "'\n";
      return -1;
    }
  }

  return 0;
}


#endif /* MULTIVIH5DATAREADER_H_ */

