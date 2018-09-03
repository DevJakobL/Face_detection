import os
import json
from wider import WIDER

# arg1: path to label
# arg2: path to images
# arg3: label file name
wider_val = WIDER(os.getcwd() + '/data/wider_face_split',
                  os.getcwd() + '/data/WIDER_val/images',
                  'wider_face_val.mat')

wider_train = WIDER(os.getcwd() + '/data/wider_face_split',
                    os.getcwd() + '/data/WIDER_train/images',
                    'wider_face_train.mat')

tdata = []

# press ctrl-C to stop the process
def cratejson(input, output):
    i = 0
    jdata = []
    for data in input.next():
        bdata = []
        for bbox in data.bboxes:
            bdata.append({"x1": bbox[0],
                          "x2": bbox[2],
                          "y1": bbox[1],
                          "y2": bbox[3]})
        if(i%3==0):
            tdata.append({"image_path": data.image_name, "rects": bdata})
        else:
            jdata.append({"image_path": data.image_name, "rects": bdata})
        i+=1

    with open(output, 'w') as f:
        f.write(json.dumps(jdata, indent=4))


def train():
    cratejson(wider_train, 'data/wider_train.json')
    print "ende train"


def val():
    cratejson(wider_val, 'data/wider_val.json')
    print "ende val"
def test():
    with open("data/wider_test.json", 'w') as f:
        f.write(json.dumps(tdata, indent=4))
        print "ende test"


train()
val()
test()
