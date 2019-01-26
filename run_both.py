import argparse
import logging
import time

import cv2
import numpy as np
import matplotlib.pyplot as plt
#matplotlib auto

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

from datetime import datetime

logger1 = logging.getLogger('TfPoseEstimator-WebCam')
logger1.setLevel(logging.DEBUG)
ch1 = logging.StreamHandler()
ch1.setLevel(logging.DEBUG)
formatter1 = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch1.setFormatter(formatter1)
logger1.addHandler(ch1)

logger2 = logging.getLogger('TfPoseEstimator-Video')
logger2.setLevel(logging.DEBUG)
ch2 = logging.StreamHandler()
ch2.setLevel(logging.DEBUG)
formatter2 = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch2.setFormatter(formatter2)
logger2.addHandler(ch2)

fps_time = 0

'''
# 코덱 설정
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
# 파일에 저장하기 위해 VideoWriter 객체를 생성
out1 = cv2.VideoWriter('./data/output1.mp4',fourcc, 30.0, (640,480))
out2 = cv2.VideoWriter('./data/output2.mp4',fourcc, 30.0, (640,480))
'''

### 1:Webcam / 2:Video
if __name__ == '__main__':
    # 1
    parser1 = argparse.ArgumentParser(description='tf-pose-estimation realtime webcam')
    parser1.add_argument('--camera', type=int, default=0)
    parser1.add_argument('--resize', type=str, default='432x368',
                        help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser1.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')
    parser1.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin')
    parser1.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    parser1.add_argument('--video', type=str, default=None)
    # 2
    parser2 = argparse.ArgumentParser(description='tf-pose-estimation Video')
    #parser2.add_argument('--camera', type=int, default=None)
    parser2.add_argument('--video', type=str, default='')
    parser2.add_argument('--resize', type=str, default='432x368',
                        help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser2.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')
    parser2.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin')
    parser2.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    parser2.add_argument('--showBG', type=bool, default=True, help='False to show skeleton only.')
    
    args1 = parser1.parse_args()
    args2 = parser2.parse_args()
    
    logger1.debug('initialization %s : %s' % (args1.model, get_graph_path(args1.model)))
    logger2.debug('initialization %s : %s' % (args2.model, get_graph_path(args2.model)))
    
    w1, h1 = model_wh(args1.resize)
    w2, h2 = model_wh(args2.resize)
    
    if w1 > 0 and h1 > 0:
        e1 = TfPoseEstimator(get_graph_path(args1.model), target_size=(w1, h1))
    else:
        e1 = TfPoseEstimator(get_graph_path(args1.model), target_size=(432, 368))
    if w2 > 0 and h2 > 0:
        e2 = TfPoseEstimator(get_graph_path(args2.model), target_size=(w2, h2))
    else:
        e2 = TfPoseEstimator(get_graph_path(args2.model), target_size=(432, 368))
    
    cap1 = cv2.VideoCapture(args1.camera)
    cap2 = cv2.VideoCapture(args2.video)
    
    if cap2.isOpened() is False:
        print("Error opening video stream or file")
    
    #data plotting
    lstX = []
    lstY = []
    period = 0
    threshold = 0.01
    
    plt.ion()
    fig = plt.figure()
    sf = fig.add_subplot(111)
    c = datetime.now()
    d = c.strftime('%M%S%f')
    d = np.float32(d) / 1000.0
    plt.xlim(d, d+100000)
    plt.ylim(0, 1)
    line1, = sf.plot('r-')
    
    while cap2.isOpened(): # 비디오가 잡히면 loop
        ret_val1, image1 = cap1.read()
        ret_val2, image2 = cap2.read()
        
        logger1.debug('image process+')
        humans1 = e1.inference(image1, resize_to_default=(w1 > 0 and h1 > 0), upsample_size=args1.resize_out_ratio)
        logger2.debug('image process+')
        humans2 = e2.inference(image2, resize_to_default=(w2 > 0 and h2 > 0), upsample_size=args2.resize_out_ratio)
        ### 2:Video) 만약 --showBG=False 이면 skeleton만 출력
        if not args2.showBG:
            image2 = np.zeros(image2.shape)
        ###
        logger1.debug('postprocess+')
        image1 = TfPoseEstimator.draw_humans(image1, humans1, imgcopy=False)
        logger2.debug('postprocess+')
        image2 = TfPoseEstimator.draw_humans(image2, humans2, imgcopy=False)
        
        ## 이미지를 파일에 저장, VideoWriter 객체에 연속적으로 저장하면 동영상이 됨.
        #out1.write(image1)
        #out2.write(image2)
        
        """
        1) 실시간으로 동영상의 점을 불러온다 (점의 좌표를 알아야함)
        2) 실시간으로 웹캠의 점을 불러온다 (점의 좌표를 알아야함)
        3) 점 간의 norm(거리)을 구한다 (scalar)
        4) 예를 들어 점이 18개로 고정되어 있다면 각 pair점 간의 norm을 전부 구하고
        5) sum 하여 그 값을 0과 1사이로 normalization 한다 ->result
        6) result를 y축 time을 x축으로 실시간 데이터 plotting
        7) result가 어떤 threshold를 넘어설때 마다 warning을 cv2.putText로 출력해준다.
        """
        
        #point찾기
        a = TfPoseEstimator.get_centers(image1, humans1, imgcopy=False)
        b = TfPoseEstimator.get_centers(image2, humans2, imgcopy=False)
        
        for i in [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]:
            webcam_points = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
            video_points = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
            dist = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
            
            try:
                webcam_points[i] = np.array(a[i])
                video_points[i] = np.array(b[i])
            except:
                if webcam_points[i] is 0:
                    continue
                else:
                    webcam_points[i] = np.array(a[i])
                if video_points[i] is 0:
                    continue
                else:
                    video_points[i] = np.array(b[i])
                
            dist[i] = np.linalg.norm(webcam_points[i] - video_points[i])
        add = np.sum(dist)
        result = add / 1000
        
        c = datetime.now()
        d = c.strftime('%M%S%f')
        d = np.float32(d) / 1000.0
        
        lstX.append(d)
        lstY.append(result)
        
        line1.set_xdata(lstX)
        line1.set_ydata(lstY)
        
        plt.draw()
        plt.pause(0.00000000001)
        print(c.strftime('%M분 %S초%f'), result)
        
        '''
        # plot 초기화
        if period < 500:
            period = period + 1
            #print(period)
        else:
            period = 0
            plt.xlim(d, d+50000)
            plt.ylim(0, 1)
            line1, = sf.plot('r-')
        '''
        
        # 임계치 조정
        if result > threshold:
            #logger1.debug('show+')
            #logger2.debug('show+')
            cv2.putText(image1, "FPS: %f" % (1.0 / (time.time() - fps_time)),
                        (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(image2, "FPS: %f" % (1.0 / (time.time() - fps_time)),
                        (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(image1, "Wrong Pose", (100, 50),  cv2.FONT_HERSHEY_SIMPLEX, 2.35, (0,0,255), 5)
            cv2.imshow('tf-pose-estimation Webcam', image1)
            cv2.imshow('tf-pose-estimation Video', image2)
            fps_time = time.time()
        else:
            #logger1.debug('show+')
            #logger2.debug('show+')
            cv2.putText(image1, "FPS: %f" % (1.0 / (time.time() - fps_time)),
                        (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(image2, "FPS: %f" % (1.0 / (time.time() - fps_time)),
                        (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.imshow('tf-pose-estimation Webcam', image1)
            cv2.imshow('tf-pose-estimation Video', image2)
            fps_time = time.time()
            
        if cv2.waitKey(1) == 27: #ESC
            break #종료

    cv2.destroyAllWindows()
logger1.debug('finished+')
logger2.debug('finished+')
