
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 22:21:49 2020

@author: Straka
"""

# =============================================================================
# """
# 1 开启TomCat
# 2 开启NGINX(cmd)
#         C:
#         cd C:\Users\Administrator\Desktop\nginx
#         start nginx
#         nginx -s reload
# 3 开启 C:\ffmpeg-4.3-win64-static\bin\MulHYZY_dealOnly-2.py
# """
# =============================================================================
dealFps = 1/2

# import gc
import time,datetime
import queue
import threading
# import cv2 as cv
import subprocess as sp
import numpy as np
# import sys
# sys.setrecursionlimit(2**31-1) # 设置栈的大小
# sys.path.append("./")
import mxnet as mx
import os
os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = "0"
os.chdir(r'C:/BabyCam')
import gluoncv as gcv
gcv.utils.check_version('0.6.0')
from gluoncv.model_zoo import get_model
print("包加载完毕")
from gluoncv.utils import try_import_cv2
cv = try_import_cv2()
nms_thresh = 0.45
classes = ['asleep','awake','crying','quilt kicked','side asleep','on stomach', 'face covered',]
classes = ['熟睡','清醒','哭闹','踢被','侧睡','趴睡', '捂被',]

dealCtx = [mx.gpu(0)] 
# dealCtx = [mx.cpu(0)]
# 模型加载
model_dir = r'C:\BabyCam\model/yolov3/yolo3_darknet53_custom_best.params'
net_name = '_'.join(('yolo3', 'darknet53', 'custom'))
net = get_model(net_name, classes=classes, ctx=dealCtx)
net.load_parameters(model_dir, ctx=dealCtx)
net.set_nms(nms_thresh=0.45, nms_topk=400)
mx.nd.waitall()
net.hybridize()
print("模型A加载完毕")

netB = get_model(net_name, classes=classes, ctx=dealCtx)
netB.load_parameters(model_dir, ctx=dealCtx)
netB.set_nms(nms_thresh=0.45, nms_topk=400)
mx.nd.waitall()
netB.hybridize()
print("模型B加载完毕")

netC = get_model(net_name, classes=classes, ctx=dealCtx)
netC.load_parameters(model_dir, ctx=dealCtx)
netC.set_nms(nms_thresh=0.45, nms_topk=400)
mx.nd.waitall()
netC.hybridize()
print("模型C加载完毕")

netD = get_model(net_name, classes=classes, ctx=dealCtx)
netD.load_parameters(model_dir, ctx=dealCtx)
netD.set_nms(nms_thresh=0.45, nms_topk=400)
mx.nd.waitall()
netC.hybridize()
print("模型D加载完毕")

# =============================================================================
# netE = get_model(net_name, classes=classes, ctx=dealCtx)
# netE.load_parameters(model_dir, ctx=dealCtx)
# netE.set_nms(nms_thresh=0.45, nms_topk=400)
# mx.nd.waitall()
# netE.hybridize()
# print("模型E加载完毕")
# =============================================================================

tempframe = mx.nd.array(cv.cvtColor(np.ones((720, 1080, 3),np.uint8)*128, cv.COLOR_BGR2RGB)).astype('uint8')
rgb_nd, scaled_frame = gcv.data.transforms.presets.yolo.transform_test(tempframe, short=416, max_size=1024)
rgb_nd = rgb_nd.as_in_context(dealCtx[0])
class_IDs, scores, bounding_boxes = net(rgb_nd)
class_IDs, scores, bounding_boxes = netB(rgb_nd)
class_IDs, scores, bounding_boxes = netC(rgb_nd)
class_IDs, scores, bounding_boxes = netD(rgb_nd)
# class_IDs, scores, bounding_boxes = netE(rgb_nd)
person = np.sum(class_IDs==0)
hat = np.sum(class_IDs==1)
scale = 1.0 * tempframe.shape[0] / scaled_frame.shape[0]
img, result = gcv.utils.viz.cv_plot_bbox(tempframe.asnumpy(), bounding_boxes[0], scores[0], class_IDs[0], class_names=net.classes, scale=scale, thresh=nms_thresh)
del img,tempframe,rgb_nd,class_IDs,scores,bounding_boxes,person,hat,scale,result
print("初始化完毕")

from tensorflow.keras.preprocessing.image import array_to_img
def showImg(frame):
    array_to_img(frame).show()
# =============================================================================
# import datetime
# cap = cv.VideoCapture(r"rtmp://0.0.0.0:1936/live/3")
# def read():
#     now_time = datetime.datetime.strftime(datetime.datetime.now(),'%H:%M:%S')
#     ret, frame = cap.read()
#     # showImg(frame)
#     print(now_time)
#     tempframe = mx.nd.array(cv.cvtColor(frame, cv.COLOR_BGR2RGB)).astype('uint8')
#     rgb_nd, scaled_frame = gcv.data.transforms.presets.yolo.transform_test(tempframe, short=416, max_size=1024)
#     rgb_nd = rgb_nd.as_in_context(dealCtx[0])
#     class_IDs, scores, bounding_boxes = net(rgb_nd)
#     person = np.sum(class_IDs==0)
#     hat = np.sum(class_IDs==1)
#     scale = 1.0 * tempframe.shape[0] / scaled_frame.shape[0]
#     img, result = gcv.utils.viz.cv_plot_bbox(tempframe.asnumpy(), bounding_boxes[0], scores[0], class_IDs[0], class_names=net.classes, scale=scale, thresh=nms_thresh)
#     showImg(img)
#     del img,tempframe,frame
#     print(result)
# read()
# cap.release
# =============================================================================

class Live(object):
    def __init__(self):
        self.fps=25
        self.frame_queueA = queue.Queue()
        self.frame_queueB = queue.Queue()
        self.frame_queueC = queue.Queue()
        self.frame_queueD = queue.Queue()
        # self.frame_queueE = queue.Queue()
        self.maxqueue = 1
        self.infoUrl=r"D:\info.html"
        self.camera_path = r"rtmp://0.0.0.0:1936/live/3"
        self.count = np.zeros(7)
        self.height=720
        self.width=1280
        self.dealTimes = 0
        self.lastShow = int(time.time())
      
        # 摄像头rtmp
    def read_frame(self):
        cap = cv.VideoCapture(self.camera_path)
        while not cap.isOpened():
            print("尝试重新连接")
            cap = cv.VideoCapture(self.camera_path)
        # Get video information
        self.fps = int(cap.get(cv.CAP_PROP_FPS))
        while self.fps==0:
            print("尝试重新连接")
            cap = cv.VideoCapture(self.camera_path)
            self.fps = int(cap.get(cv.CAP_PROP_FPS))
        # self.fps=dealFps # ===================================================
        self.width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        print("开启接收",self.camera_path)
        print(self.width,self.height,self.fps)
        cap.set(cv.CAP_PROP_BUFFERSIZE, 3); # internal buffer will now store only 3 frames
        cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc('M', 'J', 'P', 'G')) # 降低延迟
        while cap.isOpened():
            # startTime = time.time()
            ret, frame = cap.read()
            # ret, frame = cap.read()
            if ret==False:
                print("尝试重新连接")
                while ret==False:
                    cap = cv.VideoCapture(self.camera_path)
                    cap.set(cv.CAP_PROP_BUFFERSIZE, 3); # internal buffer will now store only 3 frames
                    cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc('M', 'J', 'P', 'G')) # 降低延迟
                    ret,frame = cap.read()
                print("重新连接成功")
            # put frame into queue
            # frame = cv.resize(frame, (self.w, self.h))
            # print(gc.collect())
            while self.frame_queueA.qsize()>=self.maxqueue:
                self.frame_queueA.get()
            self.frame_queueA.put(frame)
            # print('A',self.frame_queueA.qsize())
            del frame
            # tt = (1/self.fps+startTime-time.time()) # 开始时间+每帧时间-当前时间-波动
            # time.sleep(tt if tt>0 else 0)
            
            # startTime = time.time()
            ret, frame = cap.read()
            # ret, frame = cap.read()
            if ret==False:
                print("尝试重新连接")
                while ret==False:
                    cap = cv.VideoCapture(self.camera_path)
                    cap.set(cv.CAP_PROP_BUFFERSIZE, 3); # internal buffer will now store only 3 frames
                    cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc('M', 'J', 'P', 'G')) # 降低延迟
                    ret,frame = cap.read()
                print("重新连接成功")
            # put frame into queue
            # frame = cv.resize(frame, (self.w, self.h))
            # print(gc.collect())
            while self.frame_queueB.qsize()>=self.maxqueue:
                self.frame_queueB.get()
            self.frame_queueB.put(frame)
            # print('B',self.frame_queueB.qsize())
            del frame
            # tt = (1/self.fps+startTime-time.time()) # 开始时间+每帧时间-当前时间-波动
            # time.sleep(tt if tt>0 else 0)
            
            # startTime = time.time()
            # ret, frame = cap.read()
            ret, frame = cap.read()
            if ret==False:
                print("尝试重新连接")
                while ret==False:
                    cap = cv.VideoCapture(self.camera_path)
                    ret,frame = cap.read()
                print("重新连接成功")
            # put frame into queue
            # frame = cv.resize(frame, (self.w, self.h))
            # print(gc.collect())
            while self.frame_queueC.qsize()>=self.maxqueue:
                self.frame_queueC.get()
            self.frame_queueC.put(frame)
            # print('C',self.frame_queueC.qsize())
            del frame
            # tt = (1/self.fps+startTime-time.time()) # 开始时间+每帧时间-当前时间-波动
            # time.sleep(tt if tt>0 else 0)
            
            # startTime = time.time()
            # ret, frame = cap.read()
            ret, frame = cap.read()
            if ret==False:
                print("尝试重新连接")
                while ret==False:
                    cap = cv.VideoCapture(self.camera_path)
                    ret,frame = cap.read()
                print("重新连接成功")
            # put frame into queue
            # frame = cv.resize(frame, (self.w, self.h))
            # print(gc.collect())
            while self.frame_queueD.qsize()>=self.maxqueue:
                self.frame_queueD.get()
            self.frame_queueD.put(frame)
            # print('D',self.frame_queueD.qsize())
            del frame
            # tt = (1/self.fps+startTime-time.time()) # 开始时间+每帧时间-当前时间-波动
            # tt = (1/self.fps*4+startTime-time.time()) 
            # time.sleep(tt if tt>0 else 0) # 这里要缓一下 服务器没这么好 暂时设置每四帧停一下
            
# =============================================================================
#             if int(time.time())-self.lastShow>=5:
#                 print(datetime.datetime.strftime(datetime.datetime.now(),'%H:%M:%S'))
#                 ret, frame = cap.read()
#                 showImg(frame)
#                 self.lastShow = int(time.time())
# =============================================================================
# =============================================================================
#             # startTime = time.time()
#             # ret, frame = cap.read()
#             ret, frame = cap.read()
#             if ret==False:
#                 print("尝试重新连接")
#                 while ret==False:
#                     cap = cv.VideoCapture(self.camera_path)
#                     ret,frame = cap.read()
#                 print("重新连接成功")
#             # put frame into queue
#             # frame = cv.resize(frame, (self.w, self.h))
#             # print(gc.collect())
#             while self.frame_queueE.qsize()>=self.maxqueue:
#                 self.frame_queueE.get()
#             self.frame_queueE.put(frame)
#             # print('D',self.frame_queueD.qsize())
#             del frame
#             # tt = (1/self.fps+startTime-time.time()) # 开始时间+每帧时间-当前时间-波动
#             # time.sleep(tt if tt>0 else 0) # 这里要缓一下 服务器没这么好 暂时设置每四帧停一下
# =============================================================================
                        

    def dealA(self):
        print("处理A线程开始")
        while True:
            if self.frame_queueA.empty() != True:
                t1=time.time()
                frame = self.frame_queueA.get()#取出队头
                image = np.asarray(frame, dtype=np.uint8)
                del frame
                frame = mx.nd.array(cv.cvtColor(image, cv.COLOR_BGR2RGB)).astype('uint8')
                del image
                # 以上两句所用时间最多
                rgb_nd, scaled_frame = gcv.data.transforms.presets.yolo.transform_test(frame, short=416, max_size=1024)
                rgb_nd = rgb_nd.as_in_context(dealCtx[0])
                class_IDs, scores, bounding_boxes = net(rgb_nd)
                person = np.sum(class_IDs==0)
                hat = np.sum(class_IDs==1)
                scale = 1.0 * frame.shape[0] / scaled_frame.shape[0]
                img, result = gcv.utils.viz.cv_plot_bbox(frame.asnumpy(), bounding_boxes[0], scores[0], class_IDs[0], class_names=net.classes, scale=scale, thresh=nms_thresh)
                del img
                self.dealTimes += 1
                for x in result:
                    if classes.count(x)>0:
                        i = classes.index(x)
                        self.count[i]+=1
                # print('--A', result,time.time()-t1)
                

    def dealB(self):
        print("处理B线程开始")
        while True:
            if self.frame_queueB.empty() != True:
                t2=time.time()
                frame = self.frame_queueB.get()
                image = np.asarray(frame, dtype=np.uint8)
                frame = mx.nd.array(cv.cvtColor(image, cv.COLOR_BGR2RGB)).astype('uint8')
                # 以上两句所用时间最多
                rgb_nd, scaled_frame = gcv.data.transforms.presets.yolo.transform_test(frame, short=416, max_size=1024)
                rgb_nd = rgb_nd.as_in_context(dealCtx[0])
                class_IDs, scores, bounding_boxes = netB(rgb_nd)
                person = np.sum(class_IDs==0)
                hat = np.sum(class_IDs==1)
                scale = 1.0 * frame.shape[0] / scaled_frame.shape[0]
                img, result = gcv.utils.viz.cv_plot_bbox(frame.asnumpy(), bounding_boxes[0], scores[0], class_IDs[0], class_names=net.classes, scale=scale, thresh=nms_thresh)
                for x in result:
                    if classes.count(x)>0:
                        i = classes.index(x)
                        self.count[i]+=1
                self.dealTimes += 1
                # print('---B', result,time.time()-t2)
                
    def dealC(self):
        print("处理C线程开始")
        while True:
            if self.frame_queueC.empty() != True:
                t3=time.time()
                frame = self.frame_queueC.get()#取出队头
                image = np.asarray(frame, dtype=np.uint8)
                frame = mx.nd.array(cv.cvtColor(image, cv.COLOR_BGR2RGB)).astype('uint8')
                # 以上两句所用时间最多
                rgb_nd, scaled_frame = gcv.data.transforms.presets.yolo.transform_test(frame, short=416, max_size=1024)
                rgb_nd = rgb_nd.as_in_context(dealCtx[0])
                class_IDs, scores, bounding_boxes = netC(rgb_nd)
                person = np.sum(class_IDs==0)
                hat = np.sum(class_IDs==1)
                scale = 1.0 * frame.shape[0] / scaled_frame.shape[0]
                img, result = gcv.utils.viz.cv_plot_bbox(frame.asnumpy(), bounding_boxes[0], scores[0], class_IDs[0], class_names=net.classes, scale=scale, thresh=nms_thresh)

                for x in result:
                    if classes.count(x)>0:
                        i = classes.index(x)
                        self.count[i]+=1
                self.dealTimes += 1
                # print('----C', result,time.time()-t3)
                
                
    def dealD(self):
        print("处理D线程开始")
        while True:
            if self.frame_queueD.empty() != True:
                t4=time.time()
                frame = self.frame_queueD.get()#取出队头
                image = np.asarray(frame, dtype=np.uint8)
                frame = mx.nd.array(cv.cvtColor(image, cv.COLOR_BGR2RGB)).astype('uint8')
                # 以上两句所用时间最多
                rgb_nd, scaled_frame = gcv.data.transforms.presets.yolo.transform_test(frame, short=416, max_size=1024)
                rgb_nd = rgb_nd.as_in_context(dealCtx[0])
                class_IDs, scores, bounding_boxes = netD(rgb_nd)
                person = np.sum(class_IDs==0)
                hat = np.sum(class_IDs==1)
                scale = 1.0 * frame.shape[0] / scaled_frame.shape[0]
                img, result = gcv.utils.viz.cv_plot_bbox(frame.asnumpy(), bounding_boxes[0], scores[0], class_IDs[0], class_names=net.classes, scale=scale, thresh=nms_thresh)
                for x in result:
                    if classes.count(x)>0:
                        i = classes.index(x)
                        self.count[i]+=1
                self.dealTimes += 1
                # print('-----D', result,time.time()-t4)
               
# =============================================================================
#     def dealE(self):
#         print("处理E线程开始")
#         while True:
#             if self.frame_queueE.empty() != True:
#                 t4=time.time()
#                 frame = self.frame_queueE.get()#取出队头
#                 image = np.asarray(frame, dtype=np.uint8)
#                 frame = mx.nd.array(cv.cvtColor(image, cv.COLOR_BGR2RGB)).astype('uint8')
#                 # 以上两句所用时间最多
#                 rgb_nd, scaled_frame = gcv.data.transforms.presets.yolo.transform_test(frame, short=416, max_size=1024)
#                 rgb_nd = rgb_nd.as_in_context(dealCtx[0])
#                 class_IDs, scores, bounding_boxes = netE(rgb_nd)
#                 person = np.sum(class_IDs==0)
#                 hat = np.sum(class_IDs==1)
#                 scale = 1.0 * frame.shape[0] / scaled_frame.shape[0]
#                 img, result = gcv.utils.viz.cv_plot_bbox(frame.asnumpy(), bounding_boxes[0], scores[0], class_IDs[0], class_names=net.classes, scale=scale, thresh=nms_thresh)
#                 for x in result:
#                     if classes.count(x)>0:
#                         i = classes.index(x)
#                         self.count[i]+=1
#                 self.dealTimes += 1
#                 # print('-----D', result,time.time()-t4)
# =============================================================================
                
                
    def send(self):
        time.sleep(8)
        print("信息发送线程开始")
        while True:
            result=[classes[np.argmax(self.count)]] if not np.max(self.count)<=2 else []
            # 如果三秒内某一标签最大出现次数大于两次，则计入出现次数最多的标签，作为结果输出，否则为空标签
            print(self.count,result,self.dealTimes)
            with open(self.infoUrl,'w',encoding='utf8') as f:
                f.write(str(result))
            self.count = np.zeros(7)
            self.dealTimes = 0
            time.sleep(1/dealFps)
                
    def run(self):
        # 多线程处理
        threads = [
            threading.Thread(target=Live.read_frame, args=(self,)),
            threading.Thread(target=Live.dealA, args=(self,)),
            threading.Thread(target=Live.dealB, args=(self,)),
            threading.Thread(target=Live.dealC, args=(self,)),
            threading.Thread(target=Live.dealD, args=(self,)),
            # threading.Thread(target=Live.dealE, args=(self,)),
            threading.Thread(target=Live.send, args=(self,)),
        ]
        [thread.setDaemon(True) for thread in threads]
        [thread.start() for thread in threads]
        
# os.chdir(r'C:\ffmpeg-4.3-win64-static\bin') # ffmpeg所在地址
L = Live()
L.run()


def checkTime():
    print(datetime.datetime.strftime(datetime.datetime.now(),'%H:%M:%S'))
    showImg(cv.cvtColor(L.frame_queueA.get(), cv.COLOR_BGR2RGB))
