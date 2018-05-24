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
from .blocks_original import *



class BootstrapNet:
    def __init__(self, session, data_format='channels_first', batch_size=1):
        """Creates the network

        session: tf.Session
            Tensorflow session

        data_format: str
            Either 'channels_first' or 'channels_last'.
            Running on the cpu requires 'channels_last'.

        batch_size: int
            The batch size
        """
        self.session = session
        if data_format=='channels_first':
            self.placeholder_image_pair = tf.placeholder(dtype=tf.float32, shape=(batch_size,6,192,256))
            self.placeholder_image2_2 = tf.placeholder(dtype=tf.float32, shape=(batch_size,3,48,64))
        else:
            self.placeholder_image_pair = tf.placeholder(dtype=tf.float32, shape=(batch_size,192,256,6))
            self.placeholder_image2_2 = tf.placeholder(dtype=tf.float32, shape=(batch_size,48,64,3))

        with tf.variable_scope('netFlow1'):
            netFlow1_result = flow_block_demon_original(self.placeholder_image_pair, data_format=data_format )
            self.netFlow1_result = netFlow1_result
            self.predict_flow5, self.predict_conf5 = tf.split(value=netFlow1_result['predict_flowconf5'], num_or_size_splits=2, axis=1 if data_format=='channels_first' else 3)
            self.predict_flow2, self.predict_conf2 = tf.split(value=netFlow1_result['predict_flowconf2'], num_or_size_splits=2, axis=1 if data_format=='channels_first' else 3)

        with tf.variable_scope('netDM1'):
            self.netDM1_result = depthmotion_block_demon_original(
                    image_pair=self.placeholder_image_pair, 
                    image2_2=self.placeholder_image2_2, 
                    prev_flow2=self.predict_flow2, 
                    prev_flowconf2=self.netFlow1_result['predict_flowconf2'], 
                    data_format=data_format
                    )


    def eval(self, image_pair, image2_2):
        """Runs the bootstrap network
        
        image_pair: numpy.ndarray
            Array with shape [N,6,192,256] if data_format=='channels_first'
            
            Image pair in the range [-0.5, 0.5]

        image2_2: numpy.ndarray
            Second image at resolution level 2 (downsampled two times)

            The shape for data_format=='channels_first' is [1,3,48,64]

        Returns a dict with the preditions of the bootstrap net
        """
        
        fetches = {
                'predict_flow5': self.predict_flow5,
                'predict_flow2': self.predict_flow2,
                'predict_depth2': self.netDM1_result['predict_depth2'],
                'predict_normal2': self.netDM1_result['predict_normal2'],
                'predict_rotation': self.netDM1_result['predict_rotation'],
                'predict_translation': self.netDM1_result['predict_translation'],
                }
        feed_dict = {
                self.placeholder_image_pair: image_pair,
                self.placeholder_image2_2: image2_2,
                }
        return self.session.run(fetches, feed_dict=feed_dict)



class IterativeNet:
    def __init__(self, session, data_format='channels_first', batch_size=1):
        """Creates the network

        session: tf.Session
            Tensorflow session

        data_format: str
            Either 'channels_first' or 'channels_last'.
            Running on the cpu requires 'channels_last'.

        batch_size: int
            The batch size
        """
        self.session = session

        intrinsics = np.broadcast_to(np.array([[0.89115971, 1.18821287, 0.5, 0.5]]),(batch_size,4))
        self.intrinsics = tf.constant(intrinsics, dtype=tf.float32)

        if data_format == 'channels_first':
            self.placeholder_image_pair = tf.placeholder(dtype=tf.float32, shape=(batch_size,6,192,256))
            self.placeholder_image2_2 = tf.placeholder(dtype=tf.float32, shape=(batch_size,3,48,64))
            self.placeholder_depth2 = tf.placeholder(dtype=tf.float32, shape=(batch_size,1,48,64))
            self.placeholder_normal2 = tf.placeholder(dtype=tf.float32, shape=(batch_size,3,48,64))
        else:
            self.placeholder_image_pair = tf.placeholder(dtype=tf.float32, shape=(batch_size,192,256,6))
            self.placeholder_image2_2 = tf.placeholder(dtype=tf.float32, shape=(batch_size,48,64,3))
            self.placeholder_depth2 = tf.placeholder(dtype=tf.float32, shape=(batch_size,48,64,1))
            self.placeholder_normal2 = tf.placeholder(dtype=tf.float32, shape=(batch_size,48,64,3))

        self.placeholder_rotation = tf.placeholder(dtype=tf.float32, shape=(batch_size,3))
        self.placeholder_translation = tf.placeholder(dtype=tf.float32, shape=(batch_size,3))

        with tf.variable_scope('netFlow2'):
            netFlow2_result = flow_block_demon_original(
                    image_pair=self.placeholder_image_pair,
                    image2_2=self.placeholder_image2_2,
                    intrinsics=self.intrinsics,
                    prev_predictions={
                        'predict_depth2': self.placeholder_depth2,
                        'predict_normal2': self.placeholder_normal2,
                        'predict_rotation': self.placeholder_rotation,
                        'predict_translation': self.placeholder_translation,
                        },
                    data_format=data_format,
                )
            self.netFlow2_result = netFlow2_result
            self.predict_flow5, self.predict_conf5 = tf.split(value=netFlow2_result['predict_flowconf5'], num_or_size_splits=2, axis=1 if data_format=='channels_first' else 3)
            self.predict_flow2, self.predict_conf2 = tf.split(value=netFlow2_result['predict_flowconf2'], num_or_size_splits=2, axis=1 if data_format=='channels_first' else 3)

        with tf.variable_scope('netDM2'):
            self.netDM2_result = depthmotion_block_demon_original(
                    image_pair=self.placeholder_image_pair,
                    image2_2=self.placeholder_image2_2, 
                    prev_flow2=self.predict_flow2, 
                    prev_flowconf2=self.netFlow2_result['predict_flowconf2'],
                    prev_rotation=self.placeholder_rotation,
                    prev_translation=self.placeholder_translation,
                    intrinsics=self.intrinsics,
                    data_format=data_format,
                    )

    def eval(self, image_pair, image2_2, depth2, normal2, rotation, translation ):
        """Runs the iterative network
        
        image_pair: numpy.ndarray
            Array with shape [N,6,192,256] if data_format=='channels_first'
            
            Image pair in the range [-0.5, 0.5]

        image2_2: numpy.ndarray
            Second image at resolution level 2 (downsampled two times)

            The shape for data_format=='channels_first' is [1,3,48,64]

        depth2: numpy.ndarray
            Depth prediction at resolution level 2

        normal2: numpy.ndarray
            Normal prediction at resolution level 2

        rotation: numpy.ndarray
            Rotation prediction in 3 element angle axis format

        translation: numpy.ndarray
            Translation prediction

        Returns a dict with the preditions of the iterative net
        """

        fetches = {
                'predict_flow5': self.predict_flow5,
                'predict_flow2': self.predict_flow2,
                'predict_depth2': self.netDM2_result['predict_depth2'],
                'predict_normal2': self.netDM2_result['predict_normal2'],
                'predict_rotation': self.netDM2_result['predict_rotation'],
                'predict_translation': self.netDM2_result['predict_translation'],
                }
        feed_dict = {
                self.placeholder_image_pair: image_pair,
                self.placeholder_image2_2: image2_2,
                self.placeholder_depth2: depth2,
                self.placeholder_normal2: normal2,
                self.placeholder_rotation: rotation,
                self.placeholder_translation: translation,
                }
        return self.session.run(fetches, feed_dict=feed_dict)



class RefinementNet:

    def __init__(self, session, data_format='channels_first', batch_size=1):
        """Creates the network

        session: tf.Session
            Tensorflow session

        data_format: str
            Either 'channels_first' or 'channels_last'.
            Running on the cpu requires 'channels_last'.

        batch_size: int
            The batch size
        """
        self.session = session

        if data_format == 'channels_first':
            self.placeholder_image1 = tf.placeholder(dtype=tf.float32, shape=(batch_size,3,192,256))
            self.placeholder_depth2 = tf.placeholder(dtype=tf.float32, shape=(batch_size,1,48,64))
        else:
            self.placeholder_image1 = tf.placeholder(dtype=tf.float32, shape=(batch_size,192,256,3))
            self.placeholder_depth2 = tf.placeholder(dtype=tf.float32, shape=(batch_size,48,64,1))


        with tf.variable_scope('netRefine'):
            self.netRefine_result = depth_refine_block_demon_original(
                    image1=self.placeholder_image1, 
                    depthmotion_predictions={
                        'predict_depth2': self.placeholder_depth2,
                        },
                    data_format=data_format,
                    )

    def eval(self, image1, depth2):
        """Runs the refinement network
        
        image1: numpy.ndarray
            Array with the first image with shape [N,3,192,256] if data_format=='channels_first'

        depth2: numpy.ndarray
            Depth prediction at resolution level 2

        Returns a dict with the preditions of the refinement net
        """

        fetches = {
                'predict_depth0': self.netRefine_result['predict_depth0'],
                }
        feed_dict = {
                self.placeholder_image1: image1,
                self.placeholder_depth2: depth2,
                }
        return self.session.run(fetches, feed_dict=feed_dict)

