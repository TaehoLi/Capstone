import json
from collections import OrderedDict

import argparse
import logging
import time

import cv2
import numpy as np
import matplotlib.pyplot as plt

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

from datetime import datetime

logger = logging.getLogger('TfPoseEstimator-WebCam')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

fps_time = 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation realtime webcam')
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--video', type=str, default=None)
    parser.add_argument('--resize', type=str, default='432x368',
                        help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')

    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    args = parser.parse_args()

    logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
    w, h = model_wh(args.resize)
    
    if w > 0 and h > 0:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368))
        
    logger.debug('cam read+')
    cam = cv2.VideoCapture(args.camera)
    cap = cv2.VideoCapture(args.video)
    
    if cap.isOpened() is False:
        print("Error opening video stream or file")
        
    
    # data plotting
    lstX = []
    lstY1 = []
    lstY2 = []
    threshold = 0.5 # 바꿔야됨
    
    plt.ion()
    fig1 = plt.figure(num='real-time plotting1')
    sf1 = fig1.add_subplot(111)
    plt.title('Upper Body')
    plt.xticks([0, 1500000, 3000000, 4500000, 6000000])
    plt.yticks([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
    plt.axhline(y=threshold, color='r', linestyle='--', linewidth=2)
    line1, = sf1.plot([0, 6000000], [0,1], 'b-')
    
    fig2 = plt.figure(num='real-time plotting2')
    sf2 = fig2.add_subplot(111)
    plt.title('Lower Body')
    plt.xticks([0, 1500000, 3000000, 4500000, 6000000])
    plt.yticks([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
    plt.axhline(y=threshold, color='r', linestyle='--', linewidth=2)
    line2, = sf2.plot([0, 6000000], [0,1], 'b-')
    
    with open('./data/modified2_points_mobilenet_v2_large.json', encoding="utf-8") as data_file:
        data = json.load(data_file, object_pairs_hook=OrderedDict)
    num_1 = []
    for x, y in data["upper_body"]["1"]:
        num_1.append((x,y))
    num_2 = []
    for x, y in data["upper_body"]["2"]:
        num_2.append((x,y))
    num_3 = []
    for x, y in data["upper_body"]["3"]:
        num_3.append((x,y))
    num_4 = []
    for x, y in data["upper_body"]["4"]:
        num_4.append((x,y))
    num_5 = []
    for x, y in data["upper_body"]["5"]:
        num_5.append((x,y))
    num_6 = []
    for x, y in data["upper_body"]["6"]:
        num_6.append((x,y))
    num_7 = []
    for x, y in data["upper_body"]["7"]:
        num_7.append((x,y))
    num_8 = []
    for x, y in data["lower_body"]["8"]:
        num_8.append((x,y))
    num_9 = []
    for x, y in data["lower_body"]["9"]:
        num_9.append((x,y))
    num_10 = []
    for x, y in data["lower_body"]["10"]:
        num_10.append((x,y))
    num_11 = []
    for x, y in data["lower_body"]["11"]:
        num_11.append((x,y))
    num_12 = []
    for x, y in data["lower_body"]["12"]:
        num_12.append((x,y))
    num_13 = []
    for x, y in data["lower_body"]["13"]:
        num_13.append((x,y))
    
    index = 0
    while cap.isOpened():
        try:
            ret_val, image = cam.read()
            ret_val2, image2 = cap.read()

            logger.debug('image process+')
            humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)

            logger.debug('postprocess+')
            a = TfPoseEstimator.get_centers(image, humans, imgcopy=False)
            b = []
            b.append(num_1[index])
            b.append(num_2[index])
            b.append(num_3[index])
            b.append(num_4[index])
            b.append(num_5[index])
            b.append(num_6[index])
            b.append(num_7[index])
            c = []
            c.append(num_8[index])
            c.append(num_9[index])
            c.append(num_10[index])
            c.append(num_11[index])
            c.append(num_12[index])
            c.append(num_13[index])
            index += 1
            
            image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

            L2_norm1 = [] #상체
            L2_norm2 = [] #하체
            L2_nonzero1 = []
            L2_nonzero2 = []
            
            for i in range(7):
                try:
                    L2_norm1.append(np.linalg.norm(np.array(a[i+1])-np.array(b[i]), ord=2))
                except:
                    L2_norm1.append(0.0)
                    pass
                if L2_norm1[i] is not 0.0:
                    L2_nonzero1.append(L2_norm1[i])
                else:
                    pass
            for i in range(6):
                try:
                    L2_norm2.append(np.linalg.norm(np.array(a[i+8])-np.array(c[i]), ord=2))
                except:
                    L2_norm2.append(0.0)
                    pass
                if L2_norm2[i] is not 0.0:
                    L2_nonzero2.append(L2_norm2[i])
                else:
                    pass

            normalize1 = []
            normalize2 = []
            if len(L2_nonzero1) is 0:
                normalize1.append(0.0)
            elif len(L2_nonzero1) is 1:
                normalize1.append(0.0)
            elif len(L2_nonzero1) is 2:
                normalize1.append(0.0)
            else:
                for i in range(len(L2_nonzero1)):
                    normalize1.append((L2_nonzero1[i]-min(L2_nonzero1))/(max(L2_nonzero1)-min(L2_nonzero1)))
            result1 = np.sum(normalize1)/len(normalize1)
            if len(L2_nonzero2) == 0:
                normalize2.append(0.0)
            elif len(L2_nonzero2) == 1:
                normalize2.append(0.0)
            elif len(L2_nonzero2) is 2:
                normalize2.append(0.0)
            else:
                for i in range(len(L2_nonzero2)):
                    normalize2.append((L2_nonzero2[i]-min(L2_nonzero2))/(max(L2_nonzero2)-min(L2_nonzero2)))
            result2 = np.sum(normalize2)/len(normalize2)
            c = datetime.now()
            d = c.strftime('%S%f')
            d = np.float32(d) / 10.0
            lstX.append(d)
            lstY1.append(result1)
            lstY2.append(result2)
            
            print("Data point:", result1, result2)
            
            if d > 5900000: # 1분마다 플롯 초기화
                fig1.clf()
                fig1 = plt.figure(num='real-time plotting1')
                sf1 = fig1.add_subplot(111)
                plt.title('Upper Body')
                plt.xticks([0, 1500000, 3000000, 4500000, 6000000])
                plt.yticks([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
                plt.axhline(y=threshold, color='r', linestyle='--', linewidth=2)
                lstX=[]
                lstY1=[]
                line1, = sf1.plot([0, 6000000], [0,1], 'b-')
                line1.set_data(lstX, lstY1)
                
                fig2.clf()
                fig2 = plt.figure(num='real-time plotting2')
                sf2 = fig2.add_subplot(111)
                plt.title('Lower Body')
                plt.xticks([0, 1500000, 3000000, 4500000, 6000000])
                plt.yticks([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
                plt.axhline(y=threshold, color='r', linestyle='--', linewidth=2)
                lstY2=[]
                line2, = sf2.plot([0, 6000000], [0,1], 'b-')
                line2.set_data(lstX, lstY2)
            else:
                line1.set_data(lstX, lstY1)
                line2.set_data(lstX, lstY2)
                plt.show()
                plt.pause(0.0001)
            
            
            # 임계치 조정
            if (result1 or result2) > threshold:
                cv2.putText(image, "FPS: %f" % (1.0 / (time.time() - fps_time)),
                            (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(image, "Wrong Pose", (100, 50),  cv2.FONT_HERSHEY_SIMPLEX, 2.35, (0,0,255), 5)
                cv2.imshow('tf-pose-estimation Webcam', image)
                cv2.imshow('tf-pose-estimation Video', image2)
                fps_time = time.time()
            else:
                cv2.putText(image, "FPS: %f" % (1.0 / (time.time() - fps_time)),
                            (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.imshow('tf-pose-estimation Webcam', image)
                cv2.imshow('tf-pose-estimation Video', image2)
                fps_time = time.time()

            if cv2.waitKey(1) == 27: #ESC
                break #종료
            
        except:
            print("video is over")
            break
            
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'): # press Q to destroy all windows
            cv2.destroyAllWindows()
            break

logger1.debug('finished+')
logger2.debug('finished+')
