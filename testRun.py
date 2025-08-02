import cv2
import numpy as np
from threading import Thread
from Controller import Controller
from IMU import *
import time
import threading
import os
from Detect import *
from Detect1 import *
from Camera1 import Camera1
import math
from multiprocessing import Process, Queue
from Client import *
from Config import *

class CarRun:
    def __init__(self):
        self.save_dir = "/home/orangepi/SmartSave/Saving_demo/PredResult/"  # 设置保存图像的文件夹
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.cap = Camera1(0)
        self.cap.init_cameras()
        self.control=Controller()
        self.IP = '192.168.236.77'
        
        self.flag=True  
        self.grabbed = False  
        self.finded = False
                
        self.angle = 0
        self.i_diff = 0
        self.last_diff = 0

        self.coordinates = None
        self.pos = (0, 0)
        self.last_pos = (0, 0)
        self.ares=["blue_safe","red_safe"]

        self.stop_event = threading.Event()
        self.angle = 0

        self.kind_choice=0
        self.index = 1
        self.i = 1 - self.index

        self.detection_order = ["blue", "red"]
        self.Choice = ["blue_ball", "red_ball"]
        self.find_flag = False
        self.flag_ball = 1
        self.is_notfind = False  # 未找到球

    def save_frame(self, frame, filename):
        cv2.imwrite(os.path.join(self.save_dir, filename), frame)

    def  IMU_PID(self, target_angl: float):
        # logger.info(f'初始角度：{self.init_angle}') 
        kp = 0.7  # 0.75
        ki = 0.0
        kd = 0.15 #0.03
        k_speed = 0.65
        kp = kp / 100
        ki = ki / 100
        kd = kd / 100
        count = 3

        while True:
            frame = self.cap.get_frame(0)
            handle_frames(self.IP , 8080, frame)
            angle = self.control.imu_value - self.control.init_angle   
            if angle > 180:
                angle -= 360  
            elif angle < -180:
                angle += 360
            logger.info(f'Now angle:{angle}')          
            # 计算目标角
            logger.info(f'target_angl:{target_angl},init_angle:{self.control.init_angle}')
            if target_angl > 180:
                target_angl -= 360  
            elif target_angl < -180:
                target_angl += 360        
            # 计算角度差
            diffence =  angle - target_angl
            if diffence > 180:
                diffence -= 360  
            elif diffence < -180:
                diffence += 360  
            logger.info(f'Diffence:{diffence}')
            if abs(diffence) > 4: 
                p = diffence
                self.i_diff += diffence
                i = self.i_diff
                d = diffence - self.last_diff
                self.last_diff = diffence
                turn = kp*p+ki*i+kd*d
                turn = max(min(turn, 1), -1)
                # logger.info(f'转向：{turn}')
                self.control.set_motor(0, abs(k_speed*turn)+0.45, turn)
            else:
                if(count):
                    count -= 1
                else:
                    self.control.set_motor(0, 0, 0)
                    logger.info('Stop')
                    break
                    
    # 只抓选中的一个球  0 ->黄 ; 1 ->黑
    def get_highscore(self, index):
        p1 = 0.0016  # p1 = 0.0016
        p2 = 0.00058  # p2 = 0.00055
        p21 = 0.00044  # 0.0004 0.00035 改了
        d1 = 0.00026   # d1 = 0.00026
        d2 = 0.00008   # 0.00008 
        prev_deflection = 0  
        prev_target_y = 0 
        cnt=0
        last_x=0
        last_y=0
        count =0
        last_angle = 0 

        while True:
            start = time.time() 
            frame = self.cap.get_frame(0)
            cls, self.pos , img = self.detector.detect_HighBall(frame, index) 
            # cv2.putText(img, f"center:{self.pos}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            # handle_frames(self.IP , 8080, img)
            # logger.info(f'FPS:{1/(time.time()-start)}')
        
            # PID找球
            if self.pos == (0, 0):
                self.control.set_motor(0, 0.24, -0.5) # 0.23
                cnt+=1
                if cnt>130:  # 600
                    cnt=0
                    self.FindCenter()
                    logger.info('找中心')
                    self.is_notfind = True
                    break  
                logger.info('未发现')
            else:                           
                target_x, target_y = self.pos
                
                if(last_x==target_x and last_y==target_y):
                    count+=1
                    if(count>25):  # 35 
                        self.control.move(0, 0.5, -1)
                        if(target_x>320):
                            self.control.move(1, 0.4, 0.4) #右转
                            self.control.move(0, 0.6, 1)  # 0.9
                            self.control.move(-1, 0.5, 1.3) #左转  0.4                       
                        elif(target_x<=320):
                            self.control.move(-1, 0.4, 0.4) #左转
                            self.control.move(0, 0.6, 1)
                            self.control.move(1, 0.5, 1.3) #右转                           
                        count =0
                        continue

                if 420 <= target_y < 470:   # 350
                    p1 = 0.001
                    p2 = 0.00085 # 0.0009

                if target_y > 290:  # 400 390
                    logger.info('张开爪子')
                    self.control.set_servo1(136)

                if target_y <= 470:  # 500 430
                    self.grabbed = False
                    vel_forward = p1 * abs(640 - target_y)
                    vel_turn = p2 * (target_x - 300)  #310
                    deflection = target_x - 300

                    logger.info(f'{cls}位置：{self.pos}')

                    delta_deflection = deflection - prev_deflection
                    delta_target_y = target_y - prev_target_y

                    vel_forward += d1 * delta_target_y
                    vel_turn += d2 * delta_deflection

                    if abs(deflection) < 20 or abs(vel_turn) < 0.04: #25
                        logger.info(f'前进速度：{vel_forward}')
                        self.control.set_motor(0, vel_forward, 0)
                    else:
                        logger.info(f'偏转速度：{vel_turn}')
                        self.control.set_motor(0, abs(vel_turn)+0.05, math.copysign(0.5, (target_x - 300)))

                    prev_deflection = deflection
                    prev_target_y = target_y
                    last_x , last_y = target_x , target_y
                else:
                    if not self.grabbed:
                        logger.info('前进抓球')                    
                        self.control.move(0, 0.35, 0.34)  # 前进抓球 0.35
                        self.control.set_servo1(145)
                        time.sleep(0.6)
                        self.control.move(0, 0.5, -0.9) # -0.6
                        self.grabbed = True
                        # self.control.move(-0.5, 0.5, 1.45) #1.6
                        self.control.set_motor(0, 0, 0)
                        cnt=0
                        if self.isFinshed() == 1:
                            break
                        else:
                            self.control.set_servo1(95)
                            time.sleep(0.3)
                            self.control.move(0, 0.65, -0.45)
                            self.control.move(1,0.4,0.3)
                            continue

    # 只抓普通球
    def first_d(self):
        p1 = 0.0016  # p1 = 0.0016
        p2 = 0.00058  # p2 = 0.00055
        p21 = 0.00044  # 0.0004 0.00035 改了
        d1 = 0.00026   # d1 = 0.00026
        d2 = 0.00008   # 0.00008 
        prev_deflection = 0  
        prev_target_y = 0 
        cnt=0
        last_x=0
        last_y=0
        count =0
        last_angle = 0 

        while True:
            start = time.time() 
            frame = self.cap.get_frame(0)
            cls, self.pos , img = self.detector.detect_FirsBall(frame, self.index) 
            # cv2.putText(img, f"center:{self.pos}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            # handle_frames(self.IP , 8080, img)
            # logger.info(f'FPS:{1/(time.time()-start)}')
        
            # PID找球
            if self.pos == (0, 0):
                self.control.set_motor(0, 0.243, -0.5) # 0.23
                cnt+=1
                if cnt>130:  # 600
                    cnt=0
                    self.FindCenter()
                    logger.info('找中心')
                    break  # gai
                logger.info('未发现')
            else:                           
                target_x, target_y = self.pos
                
                if(last_x==target_x and last_y==target_y):
                    count+=1
                    if(count>25):  # 35 # 10
                        self.control.move(0, 0.5, -1)
                        if(target_x>320):
                            self.control.move(1, 0.4, 0.4) #右转
                            self.control.move(0, 0.6, 1)  # 0.9
                            self.control.move(-1, 0.5, 1.3) #左转  0.4                       
                        elif(target_x<=320):
                            self.control.move(-1, 0.4, 0.4) #左转
                            self.control.move(0, 0.6, 1)
                            self.control.move(1, 0.5, 1.3) #右转                           
                        count =0
                        continue

                if 420 <= target_y < 470:   # 350
                    p1 = 0.001
                    p2 = 0.00085 # 0.0009

                if target_y > 290:  # 400 390
                    logger.info('张开爪子')
                    self.control.set_servo1(136)

                if target_y <= 470:  # 500 430
                    self.grabbed = False
                    vel_forward = p1 * abs(640 - target_y)
                    vel_turn = p2 * (target_x - 300)  #310
                    deflection = target_x - 300

                    logger.info(f'{cls}位置：{self.pos}')

                    delta_deflection = deflection - prev_deflection
                    delta_target_y = target_y - prev_target_y

                    vel_forward += d1 * delta_target_y
                    vel_turn += d2 * delta_deflection

                    if abs(deflection) < 20 or abs(vel_turn) < 0.04: #25
                        logger.info(f'前进速度：{vel_forward}')
                        self.control.set_motor(0, vel_forward, 0)
                    else:
                        logger.info(f'偏转速度：{vel_turn}')
                        self.control.set_motor(0, abs(vel_turn)+0.05, math.copysign(0.5, (target_x - 300)))

                    prev_deflection = deflection
                    prev_target_y = target_y
                    last_x , last_y = target_x , target_y
                else:
                    if not self.grabbed:
                        logger.info('前进抓球')                    
                        self.control.move(0, 0.35, 0.34)  # 前进抓球 0.35
                        self.control.set_servo1(145)
                        time.sleep(0.6)
                        self.control.move(0, 0.5, -0.9) # -0.6
                        self.grabbed = True
                        # self.control.move(-0.5, 0.5, 1.45) #1.6
                        self.control.set_motor(0, 0, 0)
                        cnt=0
                        if self.First_isFinshed() == 1:
                            break
                        else:
                            self.control.set_servo1(95)
                            time.sleep(0.3)
                            self.control.move(0, 0.65, -0.45)
                            self.control.move(1,0.4,0.3)
                            continue
                    # break

    # 找所有球
    def FindBalls(self):
        p1 = 0.00148  # 0.00125 15
        p2 = 0.00045  # 0.00055
        p21 = 0.00044  # 0.0004
        d1 = 0.00025 
        d2 = 0.00008   # 0.00008 
        prev_deflection = 0  
        prev_target_y = 0 
        cnt=0
        last_x=0
        last_y=0
        count =0
        last_angle = 0

        while True:
            start = time.time()
            frame = self.cap.get_frame(0)
            cls, self.pos , img = self.detector.detect_Balls(frame, self.index)  # , self.index
            # cv2.putText(img, f"center:{self.pos}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            # handle_frames(self.IP , 8080, img)
            # logger.info(f'FPS:{1/(time.time()-start)}')
            # PID找球

            if self.pos == (0, 0):
                self.control.set_motor(0, 0.243, self.flag_ball*(-0.5)) # 0.26
                cnt+=1
                if cnt>130:
                    cnt=0
                    self.FindCenter()
                    logger.info('找中心')
                    self.is_notfind = True
                    break
                logger.info('未发现')
            else:                           
                target_x, target_y = self.pos
                
                if(last_x==target_x and last_y==target_y):
                    count+=1
                    if(count>25):  # 35 # 10
                        self.control.move(0, 0.5, -1)
                        if(target_x>320):
                            self.control.move(1, 0.4, 0.4) #右转
                            self.control.move(0, 0.6, 0.85)  # 0.9
                            self.control.move(-1, 0.5, 1.3) #左转  0.4                       
                        elif(target_x<=320):
                            self.control.move(-1, 0.4, 0.4) #左转
                            self.control.move(0, 0.6, 0.85)
                            self.control.move(1, 0.5, 1.3) #右转                           
                        count =0
                        continue

                if 420 <= target_y < 470:   # 350
                    p1 = 0.0012
                    p2 = 0.00075  # 0.0008

                if target_y > 290:  # 400 390
                    logger.info('张开爪子')
                    self.control.set_servo1(136) # 100 105

                if target_y <= 470:  #  440
                    self.grabbed = False
                    vel_forward = p1 * abs(640 - target_y)
                    vel_turn = p2 * (target_x - 300)  #310
                    deflection = target_x - 300

                    logger.info(f'{cls}位置：{self.pos}')

                    delta_deflection = deflection - prev_deflection
                    delta_target_y = target_y - prev_target_y

                    vel_forward += d1 * delta_target_y
                    vel_turn += d2 * delta_deflection

                    if abs(deflection) < 20 or abs(vel_turn) < 0.04: #25
                        logger.info(f'前进速度：{vel_forward}')
                        self.control.set_motor(0, vel_forward, 0)
                    else:
                        logger.info(f'偏转速度：{vel_turn}')
                        self.control.set_motor(0, abs(vel_turn)+0.05, math.copysign(0.5, (target_x - 300)))

                    prev_deflection = deflection
                    prev_target_y = target_y
                    last_x , last_y = target_x , target_y
                else:
                    if not self.grabbed:
                        logger.info('前进抓球')                    
                        self.control.move(0, 0.35, 0.34)  # 前进抓球 0.35
                        self.control.set_servo1(145)
                        time.sleep(0.6)
                        self.control.move(0, 0.5, -0.9) # -0.65
                        self.grabbed = True
                        # self.control.move(-0.5, 0.5, 1.45) #1.6
                        self.control.set_motor(0, 0, 0)
                        cnt=0

                        if self.isFinshed() == 1:
                            break
                        # if self.isFinshed() == 2:
                        #     continue    
                        else:
                            self.control.set_servo1(95)
                            time.sleep(0.3)
                            self.control.move(0, 0.65, -0.45)
                            self.control.move(1,0.4,0.3)
                            continue

    # 只抓黄球和黑球
    def Find_yellow_black(self):
        p1 = 0.00148  # 0.00125 15
        p2 = 0.00045  # 0.00055
        p21 = 0.00044  # 0.0004
        d1 = 0.00025 
        d2 = 0.00008   # 0.00008 
        prev_deflection = 0  
        prev_target_y = 0 
        cnt=0
        last_x=0
        last_y=0
        count =0
        last_angle = 0

        while True:
            start = time.time()
            frame = self.cap.get_frame(0)
            cls, self.pos , img = self.detector.detect_yellow_black(frame)  # , self.index
            # cv2.putText(img, f"center:{self.pos}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            # handle_frames(self.IP , 8080, img)
            # logger.info(f'FPS:{1/(time.time()-start)}')
            # PID找球

            if self.pos == (0, 0):
                self.control.set_motor(0, 0.243, (-0.5)) # 0.26
                cnt+=1
                if cnt>130:
                    cnt=0
                    self.FindCenter()
                    logger.info('找中心')
                    self.is_notfind = True
                    break
                logger.info('未发现')
            else:                           
                target_x, target_y = self.pos
                
                if(last_x==target_x and last_y==target_y):
                    count+=1
                    if(count>25):  # 35 # 10
                        self.control.move(0, 0.5, -1)
                        if(target_x>320):
                            self.control.move(1, 0.4, 0.4) #右转
                            self.control.move(0, 0.6, 0.85)  # 0.9
                            self.control.move(-1, 0.5, 1.3) #左转  0.4                       
                        elif(target_x<=320):
                            self.control.move(-1, 0.4, 0.4) #左转
                            self.control.move(0, 0.6, 0.85)
                            self.control.move(1, 0.5, 1.3) #右转                           
                        count =0
                        continue

                if 420 <= target_y < 470:   # 350
                    p1 = 0.0012
                    p2 = 0.00075  # 0.0008

                if target_y > 290:  # 400 390
                    logger.info('张开爪子')
                    self.control.set_servo1(136) # 100 105

                if target_y <= 470:  #  440
                    self.grabbed = False
                    vel_forward = p1 * abs(640 - target_y)
                    vel_turn = p2 * (target_x - 300)  #310
                    deflection = target_x - 300

                    logger.info(f'{cls}位置：{self.pos}')

                    delta_deflection = deflection - prev_deflection
                    delta_target_y = target_y - prev_target_y

                    vel_forward += d1 * delta_target_y
                    vel_turn += d2 * delta_deflection

                    if abs(deflection) < 20 or abs(vel_turn) < 0.04: #25
                        logger.info(f'前进速度：{vel_forward}')
                        self.control.set_motor(0, vel_forward, 0)
                    else:
                        logger.info(f'偏转速度：{vel_turn}')
                        self.control.set_motor(0, abs(vel_turn)+0.05, math.copysign(0.5, (target_x - 300)))

                    prev_deflection = deflection
                    prev_target_y = target_y
                    last_x , last_y = target_x , target_y
                else:
                    if not self.grabbed:
                        logger.info('前进抓球')                    
                        self.control.move(0, 0.35, 0.34)  # 前进抓球 0.35
                        self.control.set_servo1(145)
                        time.sleep(0.6)
                        self.control.move(0, 0.5, -0.9) # -0.65
                        self.grabbed = True
                        # self.control.move(-0.5, 0.5, 1.45) #1.6
                        self.control.set_motor(0, 0, 0)
                        cnt=0

                        if self.isFinshed() == 1:
                            break
                        # if self.isFinshed() == 2:
                        #     continue    
                        else:
                            self.control.set_servo1(95)
                            time.sleep(0.3)
                            self.control.move(0, 0.65, -0.45)
                            self.control.move(1,0.4,0.3)
                            continue
    # 判断第一个球是否抓住
    def First_isFinshed(self):
        while True:
            frame = self.cap.get_frame(0)
            cnt , _ = self.detector.jude_fistball(frame, self.index) 
            return cnt
                
    # 判断除黄球外的球是否抓住
    def  isFinshed(self):
        while True:
            frame = self.cap.get_frame(0)
            cnt , _ = self.detector.jude_yellow(frame, self.index)
            # cls, self.pos , img = self.detector.detect_Balls(frame, self.index)  # , self.index  detect_Balls
            return cnt

    # 找安全区
    def FindArea(self):
        p1 = 0.0068 # 0.0065
        p2 = 0.00098 #  0.00095
        p21 = 0.002
        kp1 = 0.006 # 0.0026
        kp2 = 0.001 # 0.0014
        flag = 1
        last_angle = 0
        count =0
                          
        while True:
            frame = self.cap.get_frame(0)
            cls, self.pos1 ,img = self.detector1.detect_Area(frame)
            # cv2.putText(img, f"center:{self.pos1}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            # handle_frames(self.IP , 8080, img)
            target_x, target_y = self.pos1

            angle = round(self.control.imu_value, 2)
            logger.info(f'angle:{angle} , last_angle:{last_angle}')
            if angle == last_angle:
                count+=1
                if count>16:
                    logger.info("车车已卡住，后退防卡死")
                    self.control.move(0, 0.4, -2)
                    count = 0
                    self.FindCenter()
                    logger.info('找中心')
            else:
                count=0
            last_angle = angle

            if self.pos1 == (0, 0) or cls == self.ares[1-self.index]:
                self.control.set_motor(0, 0.4, flag*(0.5))  # 0.36
            else:
                if cls == self.ares[self.index]:
                    logger.info(f'{cls}位置：{self.pos1}')
                    # if target_y >280:
                    #     logger.info(f'开始放球')
                        # self.control.set_servo1(110)
                    if target_y < 260: # 280
                        self.finded = False
                        vel_forward = (300 - target_y) * p1 if abs(300 - target_y) * p1 > 0.11 else 0.11
                        deflection = (target_x - 310) # 320
                        vel_turn = deflection * p2 

                        if abs(deflection) < 30 or abs(vel_turn) < 0.08: #or abs(vel_turn) < 0.09
                            self.control.set_motor(1, abs(vel_forward), math.copysign(0.5, vel_turn))
                            logger.info(f'前进速度：{abs(vel_forward)}')
                        else:
                            self.control.set_motor(0, abs(vel_turn), math.copysign(0.5, vel_turn))
                            logger.info(f'偏转速度：{vel_turn}')
                    else:
                        if not self.finded:
                            # if 30 < angle < 60:
                            #     logger.info('角度在(30,60)')
                            #     self.control.move(0, 0.5, -0.6)
                            #     self.IMU_PID(0)
                            #     self.control.move(0, 0.6, 0.8)
                            #     self.IMU_PID(-90)
                            #     self.control.move(0, 0.6, 0.8) #0.6s
                            # elif 120 < angle < 150:
                            #     logger.info('角度在(120,150)')
                            #     self.control.move(0, 0.5, -0.6)
                            #     self.IMU_PID(-180)
                            #     self.control.move(0, 0.6, 0.8)
                            #     self.IMU_PID(-90)
                            #     self.control.move(0, 0.65, 0.8) #0.6s
                            # else:
                            self.control.move(0, 1.6, 0.48)  # 1.1 0.55
                            self.control.set_servo1(95)
                            self.control.set_motor(0, 0, 0)
                            time.sleep(0.34)                         
                            logger.info(f'开始放球')
                            self.control.move(0, 0.75, -1.16)
                            time.sleep(0.4)
                            self.control.move(self.flag_ball*(-0.5), 0.4, 0.8) # 0.9
                            self.finded = True
                            self.is_notfind = False
                            break
                elif cls == "cross":
                    logger.info("finded cross")
                    logger.info(f'{cls}位置：{self.pos1}')
                    if target_y < 320: #380 355 340
                        vel_forward = kp1 * abs(640 - target_y)
                        vel_turn = kp2 * (target_x - 300)
                        deflection = target_x - 300
                        if abs(deflection) < 30 or abs(vel_turn) < 0.15:
                            logger.info(f'前进速度：{vel_forward}')
                            self.control.set_motor(1, vel_forward, 0)
                        else:
                            logger.info(f'偏转速度：{vel_turn}')
                            self.control.set_motor(0, abs(vel_turn), math.copysign(0.5, vel_turn))   
                    else:
                        # self.control.move(-1,0.35,0.4)
                        flag = 1

    # 找中心
    def FindCenter(self):
        kp1 = 0.0043 # 0.0046
        p2 = 0.004 # 0.002
        flag1 = 1
        last_angle = 0
        count =0

        while True:
            frame = self.cap.get_frame(0)
            cls, self.pos1 , img = self.detector1.detect_Area(frame) 
            # cv2.putText(img, f"center:{self.pos1}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            # handle_frames(self.IP , 8080, img)

            angle = round(self.control.imu_value, 2)
            logger.info(f'angle:{angle} , last_angle:{last_angle}')
            if angle == last_angle:
                count+=1
                if count>16: # 18
                    logger.info("车车已卡住，后退防卡死")
                    self.control.move(0, 0.4, -2)
                    count = 0
            else:
                count = 0
            last_angle = angle
            
            target_x, target_y = self.pos1

            if self.pos1 == (0, 0) or cls != self.ares[1-self.index]:
                self.control.set_motor(0, 0.36, flag1*(-0.5)) 
            elif cls == self.ares[1-self.index]: # self.i
                logger.info("FindCenter")
                # logger.info(f'{cls}位置：{self.pos1}')         
                if target_y < 118:  #  115
                    vel_forward = (300 - target_y) * kp1 if abs(300 - target_y) * kp1 > 0.1 else 0.1
                    vel_turn = (target_x - 320) * p2 
                    if abs(vel_turn) < 35 or abs(vel_turn) < 0.09: #or abs(vel_turn) < 0.09
                        self.control.set_motor(1, vel_forward, 0)
                        logger.info(f'前进速度：{vel_forward}')
                    else:
                        self.control.set_motor(1, abs(vel_turn), math.copysign(0.5, vel_turn)) # self.control.set_motor(0, abs(vel_turn), math.copysign(0.5, vel_turn))
                        logger.info(f'偏转速度：{vel_turn}') 
                else:
                    self.control.move(1,0.2,0.35) # move(1,0.4,0.8)
                    flag1 = -1
                    time.sleep(0.5)
                    break                 

    def run(self):
        self.detector=ObjectDetector()
        self.detector1=ObjectDetector1()

        self.control.buzzer(3)
        self.control.set_servo(90)
        self.control.set_servo1(90)      
        self.flag = False
        while True:
            dat=self.control.button()
            if dat:    
                if dat[-3]==1: 
                    self.index=0 
                    self.flag=True
                    logger.info(self.index)
                elif dat[-3]==2:
                    self.index=1
                    self.flag=True
                    logger.info(self.index)                 
            while self.flag:
                # 第一个球
                time.sleep(0.5)
                self.control.set_servo(40) # 抬起摄像头高度
                self.control.set_servo1(75)  # 接近时->148 ; 框下-> 155 ; 抬起-> 95 ; 
                self.control.move(0, 1, 1.2)  # 1.32               
                self.control.set_motor(0, 0, 0)
                time.sleep(0.5)
                self.control.move(0, 1.2, -0.65) 
                time.sleep(0.3)
                self.control.move(1,0.3,0.4)
                time.sleep(1)
                for i in range(2):
                    logger.info("寻找第一个球")
                    self.first_d()
                    logger.info('找中心')  
                    self.FindCenter()
                    logger.info("寻找安全区")
                    self.FindArea()
                    # logger.info('找中心')  
                    # self.FindCenter()
                for i in range(2):
                    self.is_notfind = False  # 未找到球
                    logger.info('寻找黄球')      
                    self.get_highscore(0)
                    logger.info('找中心')  
                    self.FindCenter()
                    if not self.is_notfind:
                        logger.info("寻找安全区")
                        self.FindArea()
                for i in range(3):
                    self.is_notfind = False  # 未找到球
                    logger.info('寻找黑球')      
                    self.Find_yellow_black()
                    logger.info('找中心')  
                    self.FindCenter()
                    if not self.is_notfind:
                        logger.info("寻找安全区")
                        self.FindArea()
                while True:
                    self.is_notfind = False  # 未找到球
                    self.flag_ball = -1
                    logger.info('寻找球')      
                    self.FindBalls()
                    logger.info('找中心')  
                    self.FindCenter()
                    if not self.is_notfind:
                        logger.info("寻找安全区")
                        self.FindArea()
                    # logger.info('找中心')  
                    # self.FindCenter()


if __name__ == "__main__":
    car = CarRun()
    car.run()  
    # car.test()
    # car.runProcessing()
