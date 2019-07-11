# Face_detection


## Installation
1. Download the wieder-face dataset from [here](http://shuoyang1213.me/WIDERFACE/)
2. Crate .json file from the wieder-face dataset to work with them in Tensorbox
    ```
    $ python wider_parser.py 
    ```
3. Install Tensorbox 
    ```
    $ cd TENSORBOX/utils && python setup.py install && ../../
    $ pip install -r path/to/requirements.txt
    ```
4. Play with Tensorbox wiht the TENSORBOX/hypes/overfeat_rezoom.json file. <a href='https://github.com/Russell91/TensorBox'> More information about Tensorbox</a> 
    ```
    $ python TENSORBOX/train.py --hypes TENSORBOX/hypes/overfeat_rezoom.json --logdir output 
    ```
5. Download trained models from [Google Drive](https://drive.google.com/open?id=1bT1g0F92nJGxxUluEP7G7hwDufyetyLp) to play with that.

     

