import argparse
import json
import os
import cv2
import tensorflow as tf
import numpy as np
from TENSORBOX.utils import train_utils
from TENSORBOX.model import TensorBox
from TENSORBOX.utils.rect import Rect
from TENSORBOX.utils.annolist import AnnotationLib as al


def cut_out_faces(H, orig_image, confidences, boxes):
    rnn_len = 1
    tau=0.25
    min_conf=0.1
    image = np.copy(orig_image[0])
    num_cells = H["grid_height"] * H["grid_width"]
    boxes_r = np.reshape(boxes, (-1,
                                 H["grid_height"],
                                 H["grid_width"],
                                 rnn_len,
                                 4))
    confidences_r = np.reshape(confidences, (-1,
                                             H["grid_height"],
                                             H["grid_width"],
                                             rnn_len,
                                             H['num_classes']))
    cell_pix_size = H['region_size']
    all_rects = [[[] for _ in range(H["grid_width"])] for _ in range(H["grid_height"])]
    for n in range(rnn_len):
        for y in range(H["grid_height"]):
            for x in range(H["grid_width"]):
                bbox = boxes_r[0, y, x, n, :]
                abs_cx = int(bbox[0]) + cell_pix_size/2 + cell_pix_size * x
                abs_cy = int(bbox[1]) + cell_pix_size/2 + cell_pix_size * y
                w = bbox[2]
                h = bbox[3]
                conf = np.max(confidences_r[0, y, x, n, 1:])
                all_rects[y][x].append(Rect(abs_cx,abs_cy,w,h,conf))
    from stitch_wrapper import stitch_rects
    acc_rects = stitch_rects(all_rects, tau)


    pairs = []
    pairs.append((acc_rects, (0, 255, 0)))
    for rect_set, color in pairs:
        for rect in rect_set:
            if rect.confidence > min_conf:
                cv2.rectangle(image,
                              (rect.cx-int(rect.width/2), rect.cy-int(rect.height/2)),
                              (rect.cx+int(rect.width/2), rect.cy+int(rect.height/2)),
                              color,
                              1)
    return image



def testimg(img_path, weights):
    hypes_file = '%s/hypes.json' % os.path.dirname(weights)
    with open(hypes_file, 'r') as f:
        H = json.load(f)
    tensorbox = TensorBox(H)
    H["grid_width"] = H["image_width"] / H["region_size"]
    H["grid_height"] = H["image_height"] / H["region_size"]
    x_in = tf.placeholder(tf.float32, name='x_in', shape=[H['image_height'], H['image_width'], 3])
    if H['use_rezoom']:
        pred_boxes, pred_logits, pred_confidences, pred_confs_deltas, pred_boxes_deltas = tensorbox.build_forward(
            tf.expand_dims(x_in, 0), 'test', reuse=None)
        grid_area = H['grid_height'] * H['grid_width']
        pred_confidences = tf.reshape(
            tf.nn.softmax(tf.reshape(pred_confs_deltas, [grid_area * H['rnn_len'], 2])),
            [grid_area, H['rnn_len'], 2])
        if H['reregress']:
            pred_boxes = pred_boxes + pred_boxes_deltas
    else:
        pred_boxes, pred_logits, pred_confidences = tensorbox.build_forward(tf.expand_dims(x_in, 0), 'test', reuse=None)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, weights)
        orig_img = cv2.imread(img_path)[:, :, :3]
        img = cv2.resize(orig_img, (640, 480))
        feed = {x_in: img}
        (np_pred_boxes, np_pred_confidences) = sess.run([pred_boxes, pred_confidences], feed_dict=feed)
        new_img = cut_out_faces(H, [img], np_pred_confidences, np_pred_boxes)
        cv2.imshow('img', new_img)
        cv2.waitKey(0)
        sess.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', required=True)
    parser.add_argument('--input_image', required=True)
    parser.add_argument('--output_dir', default='outputimages')
    args = parser.parse_args()
    testimg(args.input_image, args.weights)
