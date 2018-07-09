import argparse
import os
import cv2
import tensorflow as tf

input = ''
output = ''


def classify(img_data):
    with tf.Session() as sess:
        pass


def write_faces(faces):
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--i', required=True, type=str)
    parser.add_argument('--o', required=True, type=str)
    args = parser.parse_args()
    input = args.i
    output = args.o
    img = cv2.imread(input)
    classify(img)
    cv2.imshow('main', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
