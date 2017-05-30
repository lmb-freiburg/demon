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
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "multivih5datareader.h"


using namespace tensorflow;

REGISTER_OP("MultiViH5DataReader")
  .Attr("num_outputs: int")
  .Attr("param_json: string")
  .SetIsStateful()
  .Output("info: float")
  .Output("sample_id: int8")
  .Output("output: num_outputs * float")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) 
    {
      using namespace ::tensorflow::shape_inference;
      std::string json_str;
      c->GetAttr("param_json", &json_str);
      int num_outputs;
      c->GetAttr("num_outputs", &num_outputs);
      MultiViH5Params params;
      if( 0 != MultiViH5Params_from_json(json_str, &params) )
        return errors::InvalidArgument("cannot parse 'param_json'. See error msg in console.");

      if( int(params.top_output.size()) != num_outputs )
        return errors::InvalidArgument("num_outputs is wrong");

      DimensionHandle batch_dim = c->MakeDim(params.batch_size);
      DimensionHandle height_dim, width_dim;
      if( params.scaled_width )
        width_dim = c->MakeDim(params.scaled_width.value());
      else
        width_dim = c->UnknownDim();

      if( params.scaled_height )
        height_dim = c->MakeDim(params.scaled_height.value());
      else
        height_dim = c->UnknownDim();

      int output_idx = 0;

      // info tensor
      {
        ShapeHandle shape = c->MakeShape({5});
        c->set_output(output_idx, shape);
        ++output_idx;
      }
      // sample_ids tensor
      {
        ShapeHandle shape = c->MakeShape({c->UnknownDim()});
        c->set_output(output_idx, shape);
        ++output_idx;
      }
      if( params.top_output.count(MultiViH5Params::IMAGE_PAIR) )
      {
        ShapeHandle shape;
        if(params.convert_to_gray_values)
          shape = c->MakeShape({batch_dim, 2, height_dim, width_dim});
        else
          shape = c->MakeShape({batch_dim, 6, height_dim, width_dim});
        c->set_output(output_idx, shape);
        ++output_idx;
      }
      if( params.top_output.count(MultiViH5Params::MOTION) )
      {
        ShapeHandle shape;
        switch( params.motion_format )
        {
        case MultiViH5Params::FMATRIX:
          shape = c->MakeShape({batch_dim, 8+3}); // last element of F is 1
          break;
        case MultiViH5Params::ANGLEAXIS6:
          shape = c->MakeShape({batch_dim, 3+3});
          break;
        case MultiViH5Params::ANGLEAXIS7:
        case MultiViH5Params::QUATERNION:
          shape = c->MakeShape({batch_dim, 4+3});
          break;
        }
        c->set_output(output_idx, shape);
        ++output_idx;
      }
      if( params.top_output.count(MultiViH5Params::FLOW) )
      {
        ShapeHandle shape;
        shape = c->MakeShape({batch_dim, 2, height_dim, width_dim});
        c->set_output(output_idx, shape);
        ++output_idx;
      }
      if( params.top_output.count(MultiViH5Params::DEPTH) )
      {
        ShapeHandle shape;
        if(params.depth_pair)
          shape = c->MakeShape({batch_dim, 2, height_dim, width_dim});
        else
          shape = c->MakeShape({batch_dim, 1, height_dim, width_dim});
        c->set_output(output_idx, shape);
        ++output_idx;
      }
      if( params.top_output.count(MultiViH5Params::INTRINSICS) )
      {
        ShapeHandle shape;
        shape = c->MakeShape({batch_dim, 4});
        c->set_output(output_idx, shape);
        ++output_idx;
      }
      if( params.top_output.count(MultiViH5Params::DEPTHMASKS) )
      {
        ShapeHandle shape;
        if(params.depth_pair)
          shape = c->MakeShape({batch_dim, 2, height_dim, width_dim});
        else
          shape = c->MakeShape({batch_dim, 1, height_dim, width_dim});
        c->set_output(output_idx, shape);
        ++output_idx;
      }
      return Status::OK();
    })
  .Doc(R"doc(
Reads data in the multi view h5 format.

This op reads one or more hdf5 files and generates data samples.
The data is returned in NCHW format.


num_outputs: The number of data tensors to return.  This number depends on the
  values passed in `param_json`.

param_json:
  The parameters passed to the reader in JSON format as a string.
  It is recommended to create a python dict with all parameters first and then convert
  the dict to str with json.dumps().
  Here is an example python dict with comments and good values for training:

  ```
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

info: The info tensor stores information about the internal buffers.
  It stores the following information:
   - required number of test iterations
   - current batch buffer size
   - maximum batch buffer size
   - current reader buffer size
   - maximum reader buffer size

sample_id: A tensor storing a string with the id for each batch item.
  A newline symbol is used to separate the individual id strings.

output: A list of tensors with the requested data.

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
  E.g. a point visible in the first image maps to a point within the image borders
  of the second image.

)doc");



class MultiViH5DataReaderOp : public OpKernel 
{
public:
  explicit MultiViH5DataReaderOp(OpKernelConstruction* construction)
    :OpKernel(construction), num_test_iterations(0)
  { 
    output_data_ptr.fill(nullptr);

    OP_REQUIRES_OK(construction, construction->GetAttr("param_json", &param_json));
    MultiViH5Params_from_json(param_json, &param);
    infovec_shape.AddDim(5);
    // do initialization in Compute()
  }
 
  void Compute( OpKernelContext* context ) override 
  {
    if( !data_reader )
    {
      data_reader.reset( new MultiViH5DataReader(&param) );

      std::vector<int> imagepair;
      std::vector<int> motion;
      std::vector<int> flow;
      std::vector<int> depth;
      std::vector<int> intrinsics;
      std::vector<int> depthmasks;
      data_reader->getShape(imagepair, motion, flow, depth, intrinsics, depthmasks);


      for( int d : imagepair )
        imagepair_shape.AddDim(d);
      for( int d : depth )
        depth_shape.AddDim(d);
      for( int d : flow )
        flow_shape.AddDim(d);
      for( int d : motion )
        motion_shape.AddDim(d);
      for( int d : intrinsics )
        intrinsics_shape.AddDim(d);
      for( int d : depthmasks )
        depthmasks_shape.AddDim(d);

      num_test_iterations = data_reader->getNumberOfTestIterations();
    }

    output_tensors.clear();
    // The order is important!
    { // info tensor
      data_reader->getBufferStates(batch_buffer_state, read_buffer_state);

      Tensor* output = 0;
      OP_REQUIRES_OK(context, context->allocate_output(output_tensors.size(), infovec_shape, &output));
      float* data = output->flat<float>().data();
      output_tensors.push_back(output);
      data[0] = num_test_iterations;
      data[1] = batch_buffer_state.first;
      data[2] = batch_buffer_state.second;
      data[3] = read_buffer_state.first;
      data[4] = read_buffer_state.second;
    }
    
    output_tensors.push_back(nullptr); // placeholder for the sample_ids tensor

    if( imagepair_shape.dims() )
    {
      Tensor* output = 0;
      OP_REQUIRES_OK(context, context->allocate_output(output_tensors.size(), imagepair_shape, &output));
      output_data_ptr[0] = output->flat<float>().data();
      output_tensors.push_back(output);
    }

    if( motion_shape.dims() )
    {
      Tensor* output = 0;
      OP_REQUIRES_OK(context, context->allocate_output(output_tensors.size(), motion_shape, &output));
      output_data_ptr[1] = output->flat<float>().data();
      output_tensors.push_back(output);
    }

    if( flow_shape.dims() )
    {
      Tensor* output = 0;
      OP_REQUIRES_OK(context, context->allocate_output(output_tensors.size(), flow_shape, &output));
      output_data_ptr[2] = output->flat<float>().data();
      output_tensors.push_back(output);
    }

    if( depth_shape.dims() )
    {
      Tensor* output = 0;
      OP_REQUIRES_OK(context, context->allocate_output(output_tensors.size(), depth_shape, &output));
      output_data_ptr[3] = output->flat<float>().data();
      output_tensors.push_back(output);
    }

    if( intrinsics_shape.dims() )
    {
      Tensor* output = 0;
      OP_REQUIRES_OK(context, context->allocate_output(output_tensors.size(), intrinsics_shape, &output));
      output_data_ptr[4] = output->flat<float>().data();
      output_tensors.push_back(output);
    }

    if( depthmasks_shape.dims() )
    {
      Tensor* output = 0;
      OP_REQUIRES_OK(context, context->allocate_output(output_tensors.size(), depthmasks_shape, &output));
      output_data_ptr[5] = output->flat<float>().data();
      output_tensors.push_back(output);
    }

    {
      data_reader->getData(
          output_data_ptr[0],
          output_data_ptr[1],
          output_data_ptr[2],
          output_data_ptr[3],
          output_data_ptr[4],
          output_data_ptr[5],
          &sample_ids);

      TensorShape shape;
      shape.AddDim(sample_ids.size());
      Tensor* output = 0;
      OP_REQUIRES_OK(context, context->allocate_output(1, shape, &output));
      output_tensors[1] = output;

      int8_t* data_ptr = output->flat<int8_t>().data();
      data_ptr = std::copy(sample_ids.begin(),sample_ids.end(), data_ptr);
    }
  }

private:
  std::string param_json;
  MultiViH5Params param;
  std::unique_ptr<MultiViH5DataReader> data_reader;
  TensorShape infovec_shape;
  TensorShape imagepair_shape;
  TensorShape motion_shape;
  TensorShape flow_shape;
  TensorShape depth_shape;
  TensorShape intrinsics_shape;
  TensorShape depthmasks_shape;
  std::vector<Tensor*> output_tensors;
  std::array<float*,6> output_data_ptr; // ptrs for the getData() function
  bool get_sample_ids;
  std::string sample_ids;
  float num_test_iterations;
  std::pair<int,int> batch_buffer_state;
  std::pair<int,int> read_buffer_state;
};

REGISTER_KERNEL_BUILDER(
    Name("MultiViH5DataReader")
    .Device(DEVICE_CPU),
    MultiViH5DataReaderOp);

