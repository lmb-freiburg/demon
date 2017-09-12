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
import tensorflow as tf
import numpy as np
from tfutils import *
import os
import sys
import glob
import json

import depthmotionnet.datareader as datareader
from depthmotionnet.v2.blocks import *
from depthmotionnet.v2.losses import *

#
# global parameters
#
_num_gpus = get_gpu_count()
_num_gpus=1 # multi gpu training is not well tested

# the train dir stores all checkpoints and summaries. The dir name is the name of this file without .py
_train_dir = os.path.splitext(os.path.basename(__file__))[0]

# set the path to the training h5 files here
_data_dir = '../../datasets/traindata'


# The training procedure has several stages, which we call evolutions.
# The training parameters and the network graph change with each evolution.
_evolutions = ('0_flow1', '1_dm1', '2_flow2', '3_dm2', '4_iterative', '5_refine')
_k = 1000
_max_iter_dict = {
        '0_flow1': 1000*_k,
        '1_dm1': 1000*_k,
        '2_flow2': 250*_k,
        '3_dm2': 250*_k,
        '4_iterative': 1500*_k,
        '5_refine': 250*_k,
        }

_base_lr_dict = {
        '0_flow1': 0.00025,
        '1_dm1': 0.0002,
        '2_flow2': 0.00015,
        '3_dm2': 0.00015,
        '4_iterative': 0.00015,
        '5_refine': 0.0002,
        }

_simulated_iterations = 4
_flow_loss_weight = 0.5*1000
_flow_grad_loss_weight = 0.25*1000
_flow_conf_loss_weight = 0.5*100*0.5
_flow_conf_grad_loss_weight = 0.25*100
_depth_loss_weight = 0.5*300
_depth_grad_loss_weight = 0.25*1500
_normal_loss_weight = 0.5*50
_rotation_loss_weight = 160
_translation_loss_weight = 15*3
_kernel_regularizer = tf.contrib.layers.l2_regularizer(0.0004)


def main(argv=None):

    # Setup the session and the EvolutionTrainer.
    # The trainer object manages snapshots, multiple evolutions and provides 
    # a mainloop for training.
    gpu_options = tf.GPUOptions()
    session = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options))
    trainer = EvolutionTrainer(session, _train_dir, _evolutions)
    max_iter = _max_iter_dict[trainer.current_evo.name()]

    global_stepf = tf.to_float(trainer.global_step())

    top_output = ('IMAGE_PAIR', 'MOTION', 'DEPTH', 'INTRINSICS')

    batch_size = 32
    if trainer.current_evo >= '4_iterative':
        batch_size = 8

    reader_params = {
        'batch_size': batch_size,
        'test_phase': False,
        'motion_format': 'ANGLEAXIS6',
        'inverse_depth': True,
        'builder_threads': 1,
        'scaled_width': 256,
        'scaled_height': 192,
        'norm_trans_scale_depth': True,        
        'top_output': top_output,
        'scene_pool_size': 650,
        'builder_threads': 8,
    }

    # add data sources
    reader_params = datareader.add_sources(reader_params, glob.glob(os.path.join(_data_dir,'sun3d_train*.h5')), 0.8)
    reader_params = datareader.add_sources(reader_params, glob.glob(os.path.join(_data_dir,'rgbd_*_train.h5')), 0.2)
    reader_params = datareader.add_sources(reader_params, glob.glob(os.path.join(_data_dir,'mvs_breisach.h5')), 0.3)
    reader_params = datareader.add_sources(reader_params, glob.glob(os.path.join(_data_dir,'mvs_citywall.h5')), 0.3)
    reader_params = datareader.add_sources(reader_params, glob.glob(os.path.join(_data_dir,'mvs_achteck_turm.h5')), 0.003)
    reader_params = datareader.add_sources(reader_params, glob.glob(os.path.join(_data_dir,'scenes11_train.h5')), 0.2)

    learning_rate = ease_in_quad(
        global_stepf-_max_iter_dict[trainer.current_evo.name()]/3,
        _base_lr_dict[trainer.current_evo.name()], 
        1e-6-_base_lr_dict[trainer.current_evo.name()], 
        float(2*_max_iter_dict[trainer.current_evo.name()]/3) )
    tf.summary.scalar('LearningRate',learning_rate)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=1e-6, use_locking=False)


    with tf.name_scope("datareader"):
        reader_tensors = datareader.multi_vi_h5_data_reader(len(top_output), json.dumps(reader_params))
        data_tensors = reader_tensors[2]
        data_dict_all = dict(zip(top_output, data_tensors))
        num_test_iterations, current_batch_buffer, max_batch_buffer, current_read_buffer, max_read_buffer = tf.unstack(reader_tensors[0])
        tf.summary.scalar("datareader/batch_buffer",current_batch_buffer)
        tf.summary.scalar("datareader/read_buffer",current_read_buffer)

        # split the data for the individual towers
        data_dict_split = {}
        for k,v in data_dict_all.items():
            if k == 'INFO':
                continue # skip info vector
            if _num_gpus > 1:
                tmp = tf.split(v, num_or_size_splits=_num_gpus)
            else:
                tmp = [v]
            data_dict_split[k] = tmp

    tower_grads = []  # list of list of tuple(gradient, variable)
    tower_losses = [] # list of dict with loss_name: loss
    tower_total_losses = [] # list of tensors with the total loss for each tower

    # We use queues to collect output form the iterative part and then add this
    # output to the next mini batch.
    # Each tower uses its own queue, we store the queues and ops for all towers
    # in the following lists.
    iterative_net_queues = []
    iterative_net_queues_enqueue_ops = []
    iterative_net_queues_enqueue_ops_initialization = []
    for gpu_id in range(_num_gpus):
        with tf.device('/gpu:{0}'.format(gpu_id)), tf.name_scope('tower_{0}'.format(gpu_id)) as tower:
            reuse = gpu_id != 0
            
            data_dict = {}
            for k,v in data_dict_split.items():
                data_dict[k] = v[gpu_id]

            # dict of the losses of the current tower
            loss_dict = {}

            # data preprocessing
            with tf.name_scope("data_preprocess"):
                rotation, translation = tf.split(value=data_dict['MOTION'], num_or_size_splits=2, axis=1)
                ground_truth = prepare_ground_truth_tensors(
                    data_dict['DEPTH'],
                    rotation,
                    translation,
                    data_dict['INTRINSICS'],
                )
                image1, image2 = tf.split(value=data_dict['IMAGE_PAIR'], num_or_size_splits=2, axis=1)
                image2_2 = tf.transpose(tf.image.resize_area(tf.transpose(image2,perm=[0,2,3,1]), (48,64)), perm=[0,3,1,2])
                if trainer.current_evo >= '5_refine':
                    data_dict['image1'] = image1
                data_dict['image2_2'] = image2_2
                ground_truth['rotation'] = rotation
                ground_truth['translation'] = translation


            #
            # netFlow1
            #
            with tf.variable_scope('netFlow1', reuse=reuse):
                netFlow1_result = flow_block(
                    image_pair=data_dict['IMAGE_PAIR'], 
                    kernel_regularizer=_kernel_regularizer,
                    )
                predict_flow5, predict_conf5 = tf.split(value=netFlow1_result['predict_flowconf5'], num_or_size_splits=2, axis=1)
                predict_flow2, predict_conf2 = tf.split(value=netFlow1_result['predict_flowconf2'], num_or_size_splits=2, axis=1)

            # losses for netFlow1
            if trainer.current_evo == '0_flow1':
                with tf.name_scope('netFlow1_losses'):
                    # slowly increase the weights for the scale invariant gradient losses
                    flow_sig_weight = ease_out_quad(global_stepf, 0, _flow_grad_loss_weight, float(max_iter//3))
                    conf_sig_weight = ease_out_quad(global_stepf, 0, _flow_conf_grad_loss_weight, float(max_iter//3))
                    # slowly decrase the importance of the losses on the smaller resolution
                    level5_factor = ease_in_quad(global_stepf, 1, -1, float(max_iter//3))


                    losses = flow_loss_block(
                        gt_flow2=ground_truth['flow2'], 
                        gt_flow5=ground_truth['flow5'], 
                        gt_flow2_sig=ground_truth['flow2_sig'], 
                        pr_flow2=predict_flow2, 
                        pr_flow5=predict_flow5, 
                        pr_conf2=predict_conf2, 
                        pr_conf5=predict_conf5, 
                        flow_weight=_flow_loss_weight, 
                        conf_weight=_flow_conf_loss_weight, 
                        flow_sig_weight=flow_sig_weight,
                        conf_sig_weight=conf_sig_weight,
                        conf_diff_scale=10,
                        level5_factor=level5_factor,
                        loss_prefix='netFlow1_'
                        )
                    loss_dict.update(losses) # add to the loss dict of the current tower

                    # add selected losses to the 'losses' collection. the remaining losses are only used for summaries.
                    selected_losses = ('loss_flow5', 'loss_flow2', 'loss_flow2_sig', 'loss_conf5', 'loss_conf2', 'loss_conf2_sig')
                    for l in selected_losses:
                        tf.losses.add_loss(losses['netFlow1_'+l])


            #
            # netDM1
            #
            if trainer.current_evo >= '1_dm1':
                with tf.variable_scope('netDM1', reuse=reuse):
                    netDM1_result = depthmotion_block(
                            image_pair=data_dict['IMAGE_PAIR'],
                            image2_2=data_dict['image2_2'], 
                            prev_flow2=predict_flow2, 
                            prev_flowconf2=netFlow1_result['predict_flowconf2'], 
                            kernel_regularizer=_kernel_regularizer
                            )

            # losses for netDM1
            if trainer.current_evo in ('1_dm1',):

                with tf.name_scope('netDM1_losses'), tf.variable_scope('netDM1'):
                    # slowly increase the weights for the scale invariant gradient losses
                    depth_sig_weight = ease_out_quad(global_stepf, 0, _depth_grad_loss_weight, float(2000000))

                    losses = depthnormal_loss_block(
                        gt_depth2=ground_truth['depth2'],
                        gt_depth2_sig=ground_truth['depth2_sig'],
                        gt_normal2=ground_truth['normal2'],
                        gt_rotation=ground_truth['rotation'],
                        gt_translation=ground_truth['translation'],
                        pr_depth2=netDM1_result['predict_depth2'],
                        pr_normal2=netDM1_result['predict_normal2'],
                        pr_rotation=netDM1_result['predict_rotation'],
                        pr_translation=netDM1_result['predict_translation'],
                        depth_weight=_depth_loss_weight,
                        depth_sig_weight=depth_sig_weight, 
                        normal_weight=_normal_loss_weight, 
                        rotation_weight=_rotation_loss_weight,
                        translation_weight=_translation_loss_weight,
                        translation_factor=1,
                        loss_prefix='netDM1_',
                        )
                    loss_dict.update(losses) # add to the loss dict of the current tower

                    # add selected losses to the 'losses' collection. the remaining losses are only used for summaries.
                    selected_losses = ('loss_depth2', 'loss_depth2_sig', 'loss_normal2', 'loss_rotation', 'loss_translation')
                    for l in selected_losses:
                        tf.losses.add_loss(losses['netDM1_'+l])


            if trainer.current_evo >= '4_iterative':
                dtypes = [ v.dtype for k, v in data_dict.items() ]
                dtypes += [ v.dtype for k, v in ground_truth.items() ]
                dtypes += [ v.dtype for k, v in netDM1_result.items() ]
                names = [ k for k in data_dict ]
                names += [ k for k in ground_truth ]
                names += [ k for k in netDM1_result ]
                shapes = []
                for k,v in data_dict.items():
                    shapes.append([v.shape.as_list()[0]*(_simulated_iterations-1)] + v.shape.as_list()[1:])
                for k,v in ground_truth.items():
                    shapes.append([v.shape.as_list()[0]*(_simulated_iterations-1)] + v.shape.as_list()[1:])
                for k,v in netDM1_result.items():
                    shapes.append([v.shape.as_list()[0]*(_simulated_iterations-1)] + v.shape.as_list()[1:])

                queue = tf.FIFOQueue(
                    capacity=2,
                    dtypes=dtypes,
                    shapes=shapes,
                    names=names,
                    )
                iterative_net_queues.append(queue)

                enqueue_data_dict_initialization = {}
                for k in data_dict:
                    enqueue_data_dict_initialization[k] = tf.concat((_simulated_iterations-1)*[data_dict[k]], axis=0)
                for k in ground_truth:
                    enqueue_data_dict_initialization[k] = tf.concat((_simulated_iterations-1)*[ground_truth[k]], axis=0)
                for k in netDM1_result:
                    enqueue_data_dict_initialization[k] = tf.concat((_simulated_iterations-1)*[netDM1_result[k]], axis=0)
                #for k,v in enqueue_data_dict_initialization.items():
                    #print(k,v.shape.as_list())
                enqueue_op_initialization = queue.enqueue(enqueue_data_dict_initialization)
                iterative_net_queues_enqueue_ops_initialization.append(enqueue_op_initialization)

                data_from_queue = queue.dequeue()
                
                for k in data_dict:
                    data_dict[k] = tf.concat((data_dict[k],data_from_queue[k]),axis=0)
                for k in ground_truth:
                    ground_truth[k] = tf.concat((ground_truth[k],data_from_queue[k]),axis=0)
                for k in netDM1_result:
                    netDM1_result[k] = tf.concat((netDM1_result[k],data_from_queue[k]),axis=0)



            #
            # netFlow2
            #
            if trainer.current_evo >= '2_flow2':
                with tf.variable_scope('netFlow2', reuse=reuse):
                    netFlow2_result = flow_block(
                        image_pair=data_dict['IMAGE_PAIR'],
                        image2_2=data_dict['image2_2'],
                        intrinsics=data_dict['INTRINSICS'],
                        prev_predictions=netDM1_result,
                        kernel_regularizer=_kernel_regularizer
                        )
                    predict_flow5, predict_conf5 = tf.split(value=netFlow2_result['predict_flowconf5'], num_or_size_splits=2, axis=1)
                    predict_flow2, predict_conf2 = tf.split(value=netFlow2_result['predict_flowconf2'], num_or_size_splits=2, axis=1)

            # losses for netFlow2
            if trainer.current_evo in ('2_flow2','4_iterative'):
                with tf.name_scope('netFlow2_losses'):
                    if trainer.current_evo == '2_flow2':
                        # slowly increase the weights for the scale invariant gradient losses
                        flow_sig_weight = ease_out_quad(global_stepf, 0, _flow_grad_loss_weight, float(max_iter//3))
                        conf_sig_weight = ease_out_quad(global_stepf, 0, _flow_conf_grad_loss_weight, float(max_iter//3))
                        # slowly decrase the importance of the losses on the smaller resolution
                        level5_factor = ease_in_quad(global_stepf, 1, -1, float(max_iter//3))
                    else:
                        flow_sig_weight = _flow_grad_loss_weight
                        conf_sig_weight = _flow_conf_grad_loss_weight
                        level5_factor = 0


                    losses = flow_loss_block(
                        gt_flow2=ground_truth['flow2'], 
                        gt_flow5=ground_truth['flow5'], 
                        gt_flow2_sig=ground_truth['flow2_sig'], 
                        pr_flow2=predict_flow2, 
                        pr_flow5=predict_flow5, 
                        pr_conf2=predict_conf2, 
                        pr_conf5=predict_conf5, 
                        flow_weight=_flow_loss_weight, 
                        conf_weight=_flow_conf_loss_weight, 
                        flow_sig_weight=flow_sig_weight,
                        conf_sig_weight=conf_sig_weight,
                        conf_diff_scale=10,
                        level5_factor=level5_factor,
                        loss_prefix='netFlow2_'
                        )
                    loss_dict.update(losses) # add to the loss dict of the current tower

                    # add selected losses to the 'losses' collection. the remaining losses are only used for summaries.
                    selected_losses = ('loss_flow5', 'loss_flow2', 'loss_flow2_sig', 'loss_conf5', 'loss_conf2', 'loss_conf2_sig')
                    for l in selected_losses:
                        tf.losses.add_loss(losses['netFlow2_'+l])


            #
            # netDM2
            #
            if trainer.current_evo >= '3_dm2':
                with tf.variable_scope('netDM2', reuse=reuse):
                    netDM2_result = depthmotion_block(
                            image_pair=data_dict['IMAGE_PAIR'], 
                            image2_2=data_dict['image2_2'], 
                            prev_flow2=predict_flow2, 
                            prev_flowconf2=netFlow2_result['predict_flowconf2'], 
                            intrinsics=data_dict['INTRINSICS'],
                            prev_rotation=netDM1_result['predict_rotation'],
                            prev_translation=netDM1_result['predict_translation'],
                            kernel_regularizer=_kernel_regularizer
                            )

            # losses for netDM2
            if trainer.current_evo in ('3_dm2', '4_iterative'):

                with tf.name_scope('netDM2_losses'), tf.variable_scope('netDM2'):
                    # slowly increase the weights for the scale invariant gradient losses
                    if trainer.current_evo == '3_dm2':
                        depth_sig_weight = ease_out_quad(global_stepf, 0, _depth_grad_loss_weight, float(max_iter))
                    else:
                        depth_sig_weight = _depth_grad_loss_weight

                    losses = depthnormal_loss_block(
                        gt_depth2=ground_truth['depth2'],
                        gt_depth2_sig=ground_truth['depth2_sig'],
                        gt_normal2=ground_truth['normal2'],
                        gt_rotation=ground_truth['rotation'],
                        gt_translation=ground_truth['translation'],
                        pr_depth2=netDM2_result['predict_depth2'],
                        pr_normal2=netDM2_result['predict_normal2'],
                        pr_rotation=netDM2_result['predict_rotation'],
                        pr_translation=netDM2_result['predict_translation'],
                        depth_weight=_depth_loss_weight,
                        depth_sig_weight=depth_sig_weight, 
                        normal_weight=_normal_loss_weight, 
                        rotation_weight=_rotation_loss_weight,
                        translation_weight=_translation_loss_weight,
                        translation_factor=1,
                        loss_prefix='netDM2_',
                        )
                    loss_dict.update(losses) # add to the loss dict of the current tower

                    # add selected losses to the 'losses' collection. the remaining losses are only used for summaries.
                    selected_losses = ('loss_depth2', 'loss_depth2_sig', 'loss_normal2', 'loss_rotation', 'loss_translation')
                    for l in selected_losses:
                        tf.losses.add_loss(losses['netDM2_'+l])

            
            if trainer.current_evo == '5_refine':
                with tf.variable_scope('netRefine', reuse=reuse):
                    netRefine_result = depth_refine_block(
                            image1=data_dict['image1'], 
                            depthmotion_predictions=netDM2_result,
                            kernel_regularizer=_kernel_regularizer
                            )
                    
                with tf.name_scope('netRefine_losses'), tf.variable_scope('netRefine'):
                    # slowly increase the weights for the scale invariant gradient losses
                    depth_sig_weight = ease_out_quad(global_stepf, 0, 0.5*_depth_grad_loss_weight, float(max_iter))

                    losses = depth_refine_loss_block(
                        gt_depth0=ground_truth['depth0'],
                        gt_depth0_sig=ground_truth['depth0_sig'],
                        gt_normal0=ground_truth['normal0'],
                        pr_depth0=netRefine_result['predict_depth0'],
                        pr_normal0=netRefine_result['predict_normal0'],
                        depth_weight=_depth_loss_weight,
                        depth_sig_weight=depth_sig_weight, 
                        normal_weight=_normal_loss_weight, 
                        loss_prefix='netRefine_',
                        )
                    loss_dict.update(losses) # add to the loss dict of the current tower

                    # add selected losses to the 'losses' collection. the remaining losses are only used for summaries.
                    selected_losses = ('loss_depth0', 'loss_depth0_sig', 'loss_normal0', )
                    for l in selected_losses:
                        tf.losses.add_loss(losses['netRefine_'+l])



            
            if trainer.current_evo >= '4_iterative':
                # split the data and enqueue the 'newer' parts
                enqueue_data_dict = {}
                # dicts with data required by the next iteration
                dicts = (data_dict,ground_truth,netDM2_result) 
                for d in dicts:
                    for k in d:
                        shape = d[k].shape.as_list()
                        num =  (_simulated_iterations-1)*shape[0]//_simulated_iterations
                        slice_size = [num] + shape[1:]
                        slice_begin = len(slice_size)*[0]
                        enqueue_data_dict[k] = tf.slice(d[k], begin=slice_begin, size=slice_size)
                    
                enqueue_op = queue.enqueue(enqueue_data_dict)
                iterative_net_queues_enqueue_ops.append(enqueue_op)
                

                
            

            # generate a summary for all the individual losses in this tower
            if _num_gpus > 1:
                for name, loss in loss_dict.items():
                    tf.summary.scalar(name,loss)

            tower_losses.append(loss_dict)

            # compute loss for this tower
            losses = tf.losses.get_losses(scope=tower)
            regularization_losses = []
            if gpu_id == 0:
                regularization_losses = tf.losses.get_regularization_losses(scope=tower)
            tower_total_loss = tf.add_n(losses+regularization_losses)
            tower_total_losses.append(tower_total_loss)
            tf.summary.scalar('TotalLoss',tower_total_loss)

            # define which variables to train
            train_vars = []
            if trainer.current_evo <='0_flow1':
                train_vars.extend(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'netFlow1'))

            if trainer.current_evo in ('1_dm1', ):
                train_vars.extend(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'netDM1'))
    
            if trainer.current_evo in ('2_flow2', '4_iterative'):
                train_vars.extend(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'netFlow2'))
    
            if trainer.current_evo in ('3_dm2', '4_iterative'):
                train_vars.extend(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'netDM2'))

            if trainer.current_evo in ('5_refine',):
                train_vars.extend(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'netRefine'))
    
    
            grads_and_vars = optimizer.compute_gradients(loss=tower_total_loss, var_list=train_vars, colocate_gradients_with_ops=False)
            clipped_grads_and_vars = []
            with tf.name_scope('clip_gradients'):
                for g, v in grads_and_vars:
                    if not g is None:
                        clipped_g = tf.clip_by_value(g, clip_value_min=-100, clip_value_max=100)
                        #clipped_g = g
                        clipped_grads_and_vars.append((clipped_g,v))
                    else:
                        clipped_grads_and_vars.append((g,v))

            tower_grads.append(clipped_grads_and_vars)


    combined_losses = combine_loss_dicts(tower_losses)
    with tf.name_scope('CombinedLosses'):
        for name, loss in combined_losses.items():
            tf.summary.scalar(name, loss)
        if _num_gpus > 1:
            total_loss = tf.add_n(tower_total_losses, 'TotalLoss')
        else:
            total_loss = tower_total_losses[0]
        tf.summary.scalar('TotalLoss', total_loss)
        

    # combine gradients from all towers
    avg_grads = average_gradients(tower_grads)

    optimize_op = optimizer.apply_gradients(grads_and_vars=avg_grads, global_step=trainer.global_step())



    train_op = optimize_op

    summary_op = tf.summary.merge_all()

    train_var_summary_op_list = []
    for g, v in clipped_grads_and_vars:
        train_var_summary_op_list.append( tf.summary.histogram(v.name,  v, collections='TRAIN_VARS_SUMMARIES') )
        if not g is None:
            train_var_summary_op_list.append( tf.summary.histogram(v.name+'_grad',  g, collections='TRAIN_VARS_SUMMARIES') )
    train_var_summary_op = tf.summary.merge(train_var_summary_op_list)

    check_numeric_ops = []
    for x in train_vars:
        check_numeric_ops.append(tf.check_numerics(x, 'train var check'))
    check = tf.group(*check_numeric_ops)



    # init all vars
    init_op = tf.global_variables_initializer()
    session.run(init_op)


    # restore weights from checkpoint
    trainer.load_checkpoint()  

    if trainer.current_evo >= '4_iterative':
        # init the queues
        session.run(iterative_net_queues_enqueue_ops_initialization)
        # make sure that enqueue ops run each iteration
        train_op = tf.group(optimize_op, *iterative_net_queues_enqueue_ops)

    # define which variables to save
    save_var_dict = create_save_var_dict() # adds global_step and all trainable variables


    # train
    status = trainer.mainloop(
            max_iter=max_iter, 
            train_ops=[train_op], 
            saver_interval=100000, 
            saver_var_list=save_var_dict,
            summary_int_ops=[(_k//2,summary_op), (5*_k,train_var_summary_op)],
            display_str_ops=[('total_loss',tf.check_numerics(total_loss, 'total_loss')), ('learning_rate', learning_rate)],
            display_interval=100,
            custom_int_ops=[(1*_k,check)], 
            recovery_saver_interval=10,
            )

    trainer.coordinator().raise_requested_exception()

    return status


if __name__ == '__main__':
    tf.app.run()


