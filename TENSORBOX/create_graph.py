import tensorflow as tf
from tensorflow.python.tools.freeze_graph import freeze_graph
from utils import train_utils, googlenet_load, tf_concat
import argparse
import json
import os
from distutils.version import LooseVersion
from model import rezoom, deconv

if LooseVersion(tf.__version__) >= LooseVersion('1.0'):
    rnn_cell = tf.contrib.rnn
else:
    try:
        from tensorflow.models.rnn import rnn_cell
    except ImportError:
        rnn_cell = tf.nn.rnn_cell

def build_overfeat_inner(H, lstm_input):
    '''
    build simple overfeat decoder
    '''
    if H['rnn_len'] > 1:
        raise ValueError('rnn_len > 1 only supported with use_lstm == True')
    outputs = []
    initializer = tf.random_uniform_initializer(-0.1, 0.1)
    with tf.variable_scope('Overfeat', initializer=initializer):
        w = tf.get_variable('ip', shape=[H['later_feat_channels'], H['lstm_size']])
        outputs.append(tf.matmul(lstm_input, w))
    return outputs

def build_lstm_inner(H, lstm_input):
    '''
    build lstm decoder
    '''

    def get_lstm_cell(H):
        return rnn_cell.BasicLSTMCell(H['lstm_size'], forget_bias=0.0, state_is_tuple=True,
                                      reuse=tf.get_variable_scope().reuse)

    if H['num_lstm_layers'] > 1:
        lstm = rnn_cell.MultiRNNCell([get_lstm_cell(H) for _ in range(H['num_lstm_layers'])],
                                     state_is_tuple=True)
    else:
        lstm = get_lstm_cell(H)

    batch_size = H['batch_size'] * H['grid_height'] * H['grid_width']
    state = lstm.zero_state(batch_size, tf.float32)

    outputs = []
    with tf.variable_scope('RNN', initializer=tf.random_uniform_initializer(-0.1, 0.1)):
        for time_step in range(H['rnn_len']):
            if time_step > 0: tf.get_variable_scope().reuse_variables()
            output, state = lstm(lstm_input, state)
            outputs.append(output)
    return outputs

def build_forward(H, x, phase, reuse):
    '''
    Construct the forward model
    '''

    grid_size = H['grid_width'] * H['grid_height']
    outer_size = grid_size * H['batch_size']
    input_mean = 117.
    x -= input_mean
    cnn, early_feat = googlenet_load.model(x, H, reuse)
    early_feat_channels = H['early_feat_channels']
    early_feat = early_feat[:, :, :, :early_feat_channels]

    if H['deconv']:
        size = 3
        stride = 2
        pool_size = 5

        with tf.variable_scope("deconv", reuse=reuse):
            w = tf.get_variable('conv_pool_w',
                                shape=[size, size, H['later_feat_channels'], H['later_feat_channels']],
                                initializer=tf.random_normal_initializer(stddev=0.01))
            cnn_s = tf.nn.conv2d(cnn, w, strides=[1, stride, stride, 1], padding='SAME')
            cnn_s_pool = tf.nn.avg_pool(cnn_s[:, :, :, :256], ksize=[1, pool_size, pool_size, 1],
                                        strides=[1, 1, 1, 1], padding='SAME')

            cnn_s_with_pool = tf_concat(3, [cnn_s_pool, cnn_s[:, :, :, 256:]])
            cnn_deconv = deconv(cnn_s_with_pool,
                                output_shape=[H['batch_size'], H['grid_height'], H['grid_width'],
                                              256],
                                channels=[H['later_feat_channels'], 256])
            cnn = tf_concat(3, (cnn_deconv, cnn[:, :, :, 256:]))

    elif H['avg_pool_size'] > 1:
        pool_size = H['avg_pool_size']
        cnn1 = cnn[:, :, :, :700]
        cnn2 = cnn[:, :, :, 700:]
        cnn2 = tf.nn.avg_pool(cnn2, ksize=[1, pool_size, pool_size, 1],
                              strides=[1, 1, 1, 1], padding='SAME')
        cnn = tf_concat(3, [cnn1, cnn2])
    cnn = tf.reshape(cnn, [H['batch_size'] * H['grid_width'] * H['grid_height'],
                           H['later_feat_channels']])
    initializer = tf.random_uniform_initializer(-0.1, 0.1)
    with tf.variable_scope('decoder', reuse=reuse, initializer=initializer):
        scale_down = 0.01
        lstm_input = tf.reshape(cnn * scale_down, (H['batch_size'] * grid_size, H['later_feat_channels']))
        if H['use_lstm']:
            lstm_outputs = build_lstm_inner(H,lstm_input)
        else:
            lstm_outputs = build_overfeat_inner(H,lstm_input)

        pred_boxes = []
        pred_logits = []
        for k in range(H['rnn_len']):
            output = lstm_outputs[k]
            if phase == 'train':
                output = tf.nn.dropout(output, 0.5)
            box_weights = tf.get_variable('box_ip%d' % k,
                                          shape=(H['lstm_size'], 4))
            conf_weights = tf.get_variable('conf_ip%d' % k,
                                           shape=(H['lstm_size'], H['num_classes']))

            pred_boxes_step = tf.reshape(tf.matmul(output, box_weights) * 50,
                                         [outer_size, 1, 4])

            pred_boxes.append(pred_boxes_step)
            pred_logits.append(tf.reshape(tf.matmul(output, conf_weights),
                                          [outer_size, 1, H['num_classes']]))

        pred_boxes = tf_concat(1, pred_boxes)
        pred_logits = tf_concat(1, pred_logits)
        pred_logits_squash = tf.reshape(pred_logits,
                                        [outer_size * H['rnn_len'], H['num_classes']])
        pred_confidences_squash = tf.nn.softmax(pred_logits_squash)
        pred_confidences = tf.reshape(pred_confidences_squash,
                                      [outer_size, H['rnn_len'], H['num_classes']])

        if H['use_rezoom']:
            pred_confs_deltas = []
            pred_boxes_deltas = []
            w_offsets = H['rezoom_w_coords']
            h_offsets = H['rezoom_h_coords']
            num_offsets = len(w_offsets) * len(h_offsets)
            rezoom_features = rezoom(H, pred_boxes, early_feat, early_feat_channels, w_offsets, h_offsets)
            if phase == 'train':
                rezoom_features = tf.nn.dropout(rezoom_features, 0.5)
            for k in range(H['rnn_len']):
                delta_features = tf_concat(1, [lstm_outputs[k], rezoom_features[:, k, :] / 1000.])
                dim = 128
                delta_weights1 = tf.get_variable(
                    'delta_ip1%d' % k,
                    shape=[H['lstm_size'] + early_feat_channels * num_offsets, dim])
                # TODO: add dropout here ?
                ip1 = tf.nn.relu(tf.matmul(delta_features, delta_weights1))
                if phase == 'train':
                    ip1 = tf.nn.dropout(ip1, 0.5)
                delta_confs_weights = tf.get_variable(
                    'delta_ip2%d' % k,
                    shape=[dim, H['num_classes']])
                if H['reregress']:
                    delta_boxes_weights = tf.get_variable(
                        'delta_ip_boxes%d' % k,
                        shape=[dim, 4])
                    pred_boxes_deltas.append(tf.reshape(tf.matmul(ip1, delta_boxes_weights) * 5,
                                                        [outer_size, 1, 4]))
                scale = H.get('rezoom_conf_scale', 50)
                pred_confs_deltas.append(tf.reshape(tf.matmul(ip1, delta_confs_weights) * scale,
                                                    [outer_size, 1, H['num_classes']]))
            pred_confs_deltas = tf_concat(1, pred_confs_deltas)
            if H['reregress']:
                pred_boxes_deltas = tf_concat(1, pred_boxes_deltas)
            return pred_boxes, pred_logits, pred_confidences, pred_confs_deltas, pred_boxes_deltas

    return pred_boxes, pred_logits, pred_confidences


def create_frozen_graph(args, H):
    tf.reset_default_graph()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        freeze_graph(args.input, '', False, args.weights, "add,Reshape_2", "save/restore_all",
                     "save/Const:0", args.output, False, '')


def create_graph(args, H):
    tf.reset_default_graph()
    H["grid_width"] = H["image_width"] / H["region_size"]
    H["grid_height"] = H["image_height"] / H["region_size"]
    x_in = tf.placeholder(tf.float32, name='x_in', shape=[H['image_height'], H['image_width'], 3])
    if H['use_rezoom']:
        pred_boxes, pred_logits, pred_confidences, pred_confs_deltas, pred_boxes_deltas = build_forward(H,                                                                                                         reuse=None)
        grid_area = H['grid_height'] * H['grid_width']
        pred_confidences = tf.reshape(tf.nn.softmax(tf.reshape(pred_confs_deltas, [grid_area * H['rnn_len'], 2])),
                                      [grid_area, H['rnn_len'], 2])
        if H['reregress']:
            pred_boxes = pred_boxes + pred_boxes_deltas
    else:
        pred_boxes, pred_logits, pred_confidences = build_forward(H, tf.expand_dims(x_in, 0), 'test',
                                                                            reuse=None)
    tf.add_to_collection('placeholders', x_in)
    tf.add_to_collection('vars', pred_boxes)
    tf.add_to_collection('vars', pred_confidences)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        metafile_path = args.input if args.frozen else args.output
        metafile_path = metafile_path.split('.pb')[0]
        print(metafile_path)
        saver.save(sess, metafile_path)
        tf.train.write_graph(sess.graph.as_graph_def(), '', args.input if args.frozen else args.output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--hypes', required=True)
    parser.add_argument('--weights', required=False)
    parser.add_argument('--frozen', type=bool, default=False)
    parser.add_argument('--gpu', default=0)
    parser.add_argument('--input', default='')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    with open(args.hypes, 'r') as f:
        H = json.load(f)
    if args.frozen:
        if not os.path.exists(args.input):
            create_graph(args, H)
        create_frozen_graph(args, H)
    else:
        create_graph(args, H)
