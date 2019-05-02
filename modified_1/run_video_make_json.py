import json
from collections import OrderedDict

import argparse
import logging
import time

import cv2
import numpy as np

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

logger = logging.getLogger('TfPoseEstimator-Video')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

fps_time = 0
# 코덱 설정
fourcc = cv2.VideoWriter_fourcc(*'XVID')
# 파일에 저장하기 위해 VideoWriter 객체를 생성
out = cv2.VideoWriter('./data/skeleton.avi',fourcc, 15.0, (640,480))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation Video')
    parser.add_argument('--video', type=str, default='')
    parser.add_argument('--resize', type=str, default='0x0',
                        help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')
    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    parser.add_argument('--showBG', type=bool, default=True, help='False to show skeleton only.')
    args = parser.parse_args()

    logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
    
    w, h = model_wh(args.resize)
    if w > 0 and h > 0:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368))
            
    cap = cv2.VideoCapture(args.video)

    if cap.isOpened() is False:
        print("Error opening video stream or file")
    
    num_1 = []
    num_2 = []
    num_3 = []
    num_4 = []
    num_5 = []
    num_6 = []
    num_7 = []
    num_8 = []
    num_9 = []
    num_10 = []
    num_11 = []
    num_12 = []
    num_13 = []
    body_part = OrderedDict()
    upper = OrderedDict()
    lower = OrderedDict()
    
    while cap.isOpened():
        try:
            ret_val, image = cap.read()

            logger.debug('image process+')
            humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
            if not args.showBG:
                image = np.zeros(image.shape)

            logger.debug('postprocess+')
            point = TfPoseEstimator.get_centers(image, humans, imgcopy=False)
            image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
            #print(point)

            if len(point) == 18:
                num_1.append(point[1])
                num_2.append(point[2])
                num_3.append(point[3])
                num_4.append(point[4])
                num_5.append(point[5])
                num_6.append(point[6])
                num_7.append(point[7])
                num_8.append(point[8])
                num_9.append(point[9])
                num_10.append(point[10])
                num_11.append(point[11])
                num_12.append(point[12])
                num_13.append(point[13])


            logger.debug('show+')
            cv2.putText(image, "FPS: %f" % (1.0 / (time.time() - fps_time)), (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.imshow('tf-pose-estimation result', image)
            fps_time = time.time()

            ## 이미지를 파일에 저장, VideoWriter 객체에 연속적으로 저장하면 동영상이 됨.
            out.write(image)

            if cv2.waitKey(1) == 27:
                break
        except:
            print("video is over")
            break

    cv2.destroyAllWindows()
    
    body_part["upper_body"] = upper
    upper["1"] = num_1
    upper["2"] = num_2
    upper["3"] = num_3
    upper["4"] = num_4
    upper["5"] = num_5
    upper["6"] = num_6
    upper["7"] = num_7
        
    body_part["lower_body"] = lower
    lower["8"] = num_8
    lower["9"] = num_9
    lower["10"] = num_10
    lower["11"] = num_11
    lower["12"] = num_12
    lower["13"] = num_13
    
    # Write JSON
    with open('points.json', 'w', encoding="utf-8") as make_file:
        json.dump(body_part, make_file, ensure_ascii=False, indent="\t")
        
logger.debug('finished+')
