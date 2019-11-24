import argparse
import logging
import sys
import time

from tf_pose import common
import cv2
import numpy as np
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
import json

logger = logging.getLogger('TfPoseEstimatorRun')
logger.handlers.clear()
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation run')
    parser.add_argument('--input_path', type=str, default='input.json')
    parser.add_argument('--output_path', type=str, default='output.json')
    
    parser.add_argument('--model', type=str, default='mobilenet_thin',
                        help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    parser.add_argument('--resize', type=str, default='432x368',
                        help='if provided, resize images before they are processed. '
                             'default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')

    args = parser.parse_args()

    w, h = model_wh(args.resize)
    if w == 0 or h == 0:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368))
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))

    # estimate human poses from a single image !
#    image = common.read_imgfile(args.image, None, None)
    '''step1 init the parameters'''
    start = time.time()
    with open(args.input_path, 'r') as f:   #读取当前目录的json文件并解码成python数据
        data = json.load(f)
    image = np.array(data['img']).astype(np.uint8)

    if image is None:
        logger.error('Image can not be read, path=%s' % args.input_path)
        sys.exit(-1)

    t = time.time()
    humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
    elapsed = time.time() - t

    logger.info('inference image: %s in %.4f seconds.' % (args.input_path, elapsed))

    image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

    '''step3 output the json file'''
    data = {'img':image.tolist()}
    with open(args.output_path,"w") as f:
       json.dump(data,f)
    cv2.imwrite('frame.jpg',image)
    duration = time.time() - start
    print('processing time is',duration)

