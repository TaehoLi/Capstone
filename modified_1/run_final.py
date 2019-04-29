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


# 코덱 설정
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
# 파일에 저장하기 위해 VideoWriter 객체를 생성
#out1 = cv2.VideoWriter('./data/output3.avi', fourcc, 20.0, (640,480))
#out2 = cv2.VideoWriter('./data/output4.avi', fourcc, 20.0, (640,480))


### 1:Webcam / 2:Video
if __name__ == '__main__':
    # 1
    parser1 = argparse.ArgumentParser(description='tf-pose-estimation realtime webcam')
    parser1.add_argument('--camera', type=int, default=0)
    parser1.add_argument('--resize', type=str, default='432x368',
                        help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser1.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')
    parser1.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    parser1.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    parser1.add_argument('--video', type=str, default=None)
    # 2
    parser2 = argparse.ArgumentParser(description='tf-pose-estimation Video')
    parser2.add_argument('--camera', type=int, default=None)
    parser2.add_argument('--video', type=str, default='')
    parser2.add_argument('--resize', type=str, default='432x368',
                        help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser2.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')
    parser2.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
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
    
    
    # data plotting
    lstX = []
    lstY = []
    threshold = 0.5
    
    plt.ion()
    fig = plt.figure(num='real-time plotting')
    sf = fig.add_subplot(111)
    plt.title('stream data')
    plt.xticks([0, 1500000, 3000000, 4500000, 6000000])
    plt.yticks([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
    plt.axhline(y=threshold, color='r', linestyle='--', linewidth=2)
    line1, = sf.plot([0, 6000000], [0,1], 'b-')
    
    
    while cap2.isOpened(): # 비디오가 잡히면 loop
        try:
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


            """
            1) 실시간으로 동영상의 점을 불러온다 (점의 좌표를 알아야함)
            2) 실시간으로 웹캠의 점을 불러온다 (점의 좌표를 알아야함)
            3) 점 간의 norm(거리)을 구한다 (scalar)
            4) 예를 들어 점이 18개로 고정되어 있다면 각 pair점 간의 norm을 전부 구하고
            5) sum 하여 그 값을 0과 1사이로 normalization 한다 ->result
            6) result를 y축 time을 x축으로 실시간 데이터 plotting
            7) result가 어떤 threshold를 넘어설때 마다 warning을 cv2.putText로 출력해준다.
            """

            
            # point 찾기
            a = TfPoseEstimator.get_centers(image1, humans1, imgcopy=False)
            b = TfPoseEstimator.get_centers(image2, humans2, imgcopy=False)

            L2_norm = []
            L2_nonzero = []
            for i in range(len(b)):
                try:
                    L2_norm.append(np.linalg.norm(np.array(a[i])-np.array(b[i]), ord=2))
                except:
                    L2_norm.append(0.0)
                    pass
                if L2_norm[i] is not 0.0:
                    L2_nonzero.append(L2_norm[i])
                else:
                    pass
            
            normalize = []
            for i in range(len(L2_nonzero)):
                normalize.append((L2_nonzero[i]-min(L2_nonzero))/(max(L2_nonzero)-min(L2_nonzero)))
            normalize.append(0.0)
            #print(normalize)
            result = np.sum(normalize)/len(normalize)
            #print(result)

            c = datetime.now()
            d = c.strftime('%S%f')
            d = np.float32(d) / 10.0
            
            lstX.append(d)
            lstY.append(result)

            print("Data point:", result)
            
            if d > 5900000: # 1분마다 플롯 초기화
                fig.clf()
                sf = fig.add_subplot(111)
                plt.title('stream data')
                plt.xticks([0, 1500000, 3000000, 4500000, 6000000])
                plt.yticks([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
                plt.axhline(y=threshold, color='r', linestyle='--', linewidth=2)
                lstX=[]
                lstY=[]
                line1, = sf.plot([0, 6000000], [0,1], 'b-')
                line1.set_data(lstX, lstY)
            else:
                line1.set_data(lstX, lstY)
                plt.show()
                plt.pause(0.0001) # must need in windows display
            
            
            # 임계치 조정
            if result > threshold:
                #logger1.debug('show+')
                #logger2.debug('show+')
                cv2.putText(image1, "FPS: %f" % (1.0 / (time.time() - fps_time)),
                            (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(image2, "FPS: %f" % (1.0 / (time.time() - fps_time)),
                            (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(image1, "Wrong Pose", (100, 50),
cv2.FONT_HERSHEY_SIMPLEX, 2.35, (0,0,255), 5)
                cv2.imshow('tf-pose-estimation Webcam', image1)
                cv2.imshow('tf-pose-estimation Video', image2)
                ## 이미지를 파일에 저장, VideoWriter 객체에 연속적으로 저장하면 동영상이 됨.
                #out1.write(image1)
                #out2.write(image2)
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
                ## 이미지를 파일에 저장, VideoWriter 객체에 연속적으로 저장하면 동영상이 됨.
                #out1.write(image1)
                #out2.write(image2)
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
        
    #cv2.destroyAllWindows()
    
logger1.debug('finished+')
logger2.debug('finished+')
