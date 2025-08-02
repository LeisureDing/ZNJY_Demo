import cv2
import numpy as np
from threading import Thread
from Controller import Controller
from IMU import *
import time
import threading
import os
from Handler import *
from Detect import *
from Detect1 import *
import math
from multiprocessing import Process, Queue
import multiprocessing

class CarRun:
    def __init__(self):
        self.save_dir = "/home/orangepi/SmartSave/Saving_demo/PredResult/"  # 设置保存图像的文件夹
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.cap = cv2.VideoCapture(0)
        self.control=Controller()
        
        pygame.init()
        self.handler=HandleController()
        self.flag=True  
        self.grabbed = False  
        self.finded = False
                
        self.ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=0.5)
        datahex = self.ser.read(33)
        self.init_angle=DueData(datahex)
        self.angle = 0
        self.i_diff = 0
        self.last_diff = 0

        self.coordinates = None
        self.pos = (0, 0)
        self.last_pos = (0, 0)
        self.ares=["blue_safe","red_safe"]

        self.stop_event = threading.Event()
        self.angle = 0
        self.angle_queue = Queue()

        self.kind_choice=0
        self.index = 0

        self.ball = {
            "blue": (np.array([100, 150, 50]), np.array([140, 255, 255]), 500),
            "red": (np.array([0, 148, 0]), np.array([255, 194, 255]), 500)
        }
        self.structuring_element = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        self.detection_order = ["blue", "red"]
        self.Choice = ["blue_ball", "red_ball"]
        self.find_flag=False
        if not self.cap.isOpened():
            print("Unable to open camera")

    def save_frame(self, frame, filename):
        cv2.imwrite(os.path.join(self.save_dir, filename), frame)

    def check_handle(self):
        while self.flag:
            for event in pygame.event.get():
                if event.type == pygame.JOYBUTTONDOWN:
                    if event.button == 0:  # A 
                        self.flag=False
                        return None
        
    def read_angle(self):
        while True:
            datahex = self.ser.read(33)
            while self.ser.in_waiting > 33:
                datahex = self.ser.read(33)
            angle = DueData(datahex)
            angle = angle - self.init_angle
            if angle > 180:
                angle -= 360
            elif angle < -180:
                angle += 360

    def isCircular(self, contour, min_area):
        area = cv2.contourArea(contour)
        if area < 120:
            return False
        (x, y), r = cv2.minEnclosingCircle(contour)
        cir_area = np.pi * (r ** 2)
        cir = area / cir_area if cir_area > 0 else 0
        return cir > 0.6  # 0.65

    def cv_Process(self, frame, index):
        # 根据 index 选择要检测的球的颜色
        ball_color = list(self.ball.keys())[index]
        lower, upper, min_area = self.ball[ball_color]
        # 根据选择的颜色转换颜色空间
        if ball_color == "red":
            dem = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)
        else:
            dem = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)    
        # 应用颜色范围
        mask = cv2.inRange(dem, lower, upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.structuring_element)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.structuring_element)
        # 查找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_area = 0
        max_area_center = None
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > max_area and area > min_area and 120 < area < 8000:
                if self.isCircular(contour, min_area):
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        max_area = area
                        max_area_center = (cX, cY)
        # 绘制检测结果
        if max_area_center:
            cv2.line(frame, (max_area_center[0] - 10, max_area_center[1]), (max_area_center[0] + 10, max_area_center[1]), (0, 255, 0), 2)
            cv2.line(frame, (max_area_center[0], max_area_center[1] - 10), (max_area_center[0], max_area_center[1] + 10), (0, 255, 0), 2)
            self.coordinates = max_area_center
        else:
            self.coordinates = (0, 0)       
        return frame, self.coordinates

    def IMU_PID(self, target_angl: float):
        # print(f'初始角度：{self.init_angle}') 
        kp = 0.75
        ki = 0.0
        kd = 0.15 #0.03
        k_speed = 0.65

        kp = kp / 100
        ki = ki / 100
        kd = kd / 100

        count = 3

        # target_angl = - target_angl
        while True:
            datahex = self.ser.read(33)
            while (self.ser.in_waiting > 33):
                datahex = self.ser.read(33)
            angle = DueData(datahex)
            # print(angle)
            angle = angle - self.init_angle  
            if angle > 180:
                angle -= 360  
            elif angle < -180:
                angle += 360
            # print(f'现在的角度：{angle}')          
            # 计算目标角
            # print(f'目标角度:{target_angl},初始角度:{self.init_angle}')
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
            print(f'目标与初始角度之差：{diffence}')
            if abs(diffence) > 4: 
                p = diffence
                self.i_diff += diffence
                i = self.i_diff
                d = diffence - self.last_diff
                self.last_diff = diffence
                turn = kp*p+ki*i+kd*d
                turn = max(min(turn, 1), -1)
                # print(f'转向：{turn}')
                self.control.set_motor(0, abs(k_speed*turn)+0.45, turn)
            else:
                if(count):
                    count -= 1
                else:
                    self.control.set_motor(0, 0, 0)
                    print('停止')
                    break

    def straight(self):
        diff = self.IMU_PID(0)
        angle = 0.1 * diff
        self.control.set_motor(0,1,angle)

    def process_image(self, frame,index):
        detected_balls = []
        color_p = ["blue_ball", "red_ball"]
        color_priority = ["yellow_ball", "black_ball", color_p[index]]
        results = self.model1(frame, verbose=False)
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())
                if conf > 0.8:
                    class_name = self.class_names[cls] if cls < len(self.class_names) else "Unknown"
                    center_x = (xyxy[0] + xyxy[2]) / 2
                    center_y = (xyxy[1] + xyxy[3]) / 2
                    detected_balls.append((class_name, center_x, center_y, conf))

        # 将检测到的球转换为 NumPy 数组，加快操作
        detected_balls_array = np.array(detected_balls, dtype=object)
        # 找最优先的球
        for color in color_priority:
            # 正确地创建掩码来过滤相同颜色的球
            color_mask = [item[0] == color for item in detected_balls_array]
            if any(color_mask):
                # 获取属于当前颜色的所有球
                color_balls = np.array([item for item in detected_balls_array if item[0] == color])
                # 找到 y 坐标最大的球
                max_y_index = np.argmax([item[2] for item in color_balls])
                cur_ball, cur_ballX, cur_ballY, _ = color_balls[max_y_index]
                # print(f"{cur_ball}: {(cur_ballX, cur_ballY)}")
                cv2.circle(frame, (int(cur_ballX), int(cur_ballY)), 10, (0, 255, 0), 2)
                cv2.putText(frame, f"{cur_ball}", (int(cur_ballX), int(cur_ballY) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                self.coordinates = (cur_ballX, cur_ballY)
                return frame, self.coordinates

        self.coordinates = (0, 0)
        return frame, self.coordinates
    
    def first_FindBall(self,index):
        while True:
            _, frame = self.cap.read()
            frame_with_detection,self.pos=self.cv_Process(frame,index)
            # timestamp = int(time.time())
            # self.save_frame(frame_with_detection, f"frame_{timestamp}.jpg")
            # PID找球
            if self.pos == (0, 0):
                self.control.set_motor(0, 0.35, -0.5)
            else:
                target_x, target_y = self.pos
                if target_y < 425:  #390
                    vel_forward = abs(480 - target_y) * 0.003
                    deflection = (target_x - 300)
                    vel_turn = (target_x - 300) * 0.0006
                    # print('位置：', self.pos)                 
                    if abs(deflection) < 20 or abs(vel_turn) < 0.05:
                        self.control.set_motor(0, vel_forward, 0) 
                        # print('前进速度：', vel_forward)
                    else:
                        self.control.set_motor(0, vel_forward, vel_turn) 
                        # print('偏转速度：', vel_turn)
                else:
                    self.control.set_servo1(115)
                    self.control.set_servo2(65)
                    self.control.move(0, 0.5, 0.4) # 0.2s
                    self.control.set_servo1(130) 
                    self.control.set_servo2(55)
                    break

    # def first_d(self):
    #     p1 = 0.0013  
    #     p2 = 0.0012  # 0.0012
    #     d1 = 0.00025   #  0.0001
    #     d2 = 0.00025   #  0.0001
    #     prev_deflection = 0  
    #     prev_target_y = 0  

    #     while True:
    #         _, frame = self.cap.read()
    #         cls, self.pos = self.detector.detect_FirsBall(frame, self.index)

    #         if self.pos == (0, 0):
    #             self.control.set_motor(0, 0.4, -0.5)
    #         else:
    #             target_x, target_y = self.pos
    #             if target_y < 380:  # 430
    #                 self.grabbed = False
    #                 vel_forward = p1 * abs(640 - target_y)
    #                 vel_turn = p2 * (target_x - 300)
    #                 deflection = target_x - 300

    #                 print(f'{cls}位置：', self.pos)

    #                 delta_deflection = deflection - prev_deflection
    #                 delta_target_y = target_y - prev_target_y

    #                 vel_forward += d1 * delta_target_y
    #                 vel_turn += d2 * delta_deflection

    #                 if abs(deflection) < 35 or abs(vel_turn) < 0.1: # or abs(vel_turn) < 0.15
    #                     print('前进速度：', vel_forward)
    #                     self.control.set_motor(0, vel_forward, 0)
    #                 else:
    #                     print('偏转速度：', vel_turn)
    #                     self.control.set_motor(0, abs(vel_turn), math.copysign(0.5, vel_turn))

    #                 prev_deflection = deflection
    #                 prev_target_y = target_y
    #             else:
    #                 self.first_FindBall(self.index)
    #                 break

    def first_d(self):
        p1 = 0.0013  # 0.0013
        p2 = 0.0015  # 0.0012 0.00135
        p21 = 0.0015  # 0.0016
        d1 = 0.0002  #  0.0001 0.00025 
        d2 = 0.00025   #  0.0001 0.0002
        prev_deflection = 0  
        prev_target_y = 0 

        while True:
            # ti=time.time()
            _, frame = self.cap.read()
            cls, self.pos = self.detector.detect_FirsBall(frame, self.index) # ,save_path=f'/home/orangepi/SmartSave/Saving_demo/PredResult/img{ti}.jpg'

            # PID找球
            if self.pos == (0, 0):
                self.control.set_motor(0, 0.3, -0.5)
                print('未发现')
            else:                           
                target_x, target_y = self.pos
                if target_y > 365: #366 362
                        # while(abs(target_x - 300)>10):
                        #     print('近距离调中心')
                        #     _, frame = self.cap.read()
                        #     cls, self.pos = self.detector.detect_FirsBall(frame, self.index)
                        #     target_x, _ = self.pos
                        #     vel_turn = p21 * (target_x - 300)
                        #     print(f'中心坐标：{self.pos}')
                        #     print(f'偏转速度：{vel_turn}')
                        #     self.control.set_motor(0, abs(vel_turn), math.copysign(0.5, vel_turn))
                        print('张开爪子')
                        self.control.set_servo1(106) #90
                        self.control.set_servo2(74)
                if target_y < 382:  # 430 385
                    self.grabbed = False
                    vel_forward = p1 * abs(640 - target_y)
                    vel_turn = p2 * (target_x - 300)
                    deflection = target_x - 300

                    print(f'{cls}位置：', self.pos)

                    delta_deflection = deflection - prev_deflection
                    delta_target_y = target_y - prev_target_y

                    vel_forward += d1 * delta_target_y
                    vel_turn += d2 * delta_deflection

                    if abs(deflection) < 20 or abs(vel_turn) < 0.075: # or abs(vel_turn) < 0.1 0.15 0.07
                        print('前进速度：', vel_forward)
                        self.control.set_motor(0, vel_forward, 0)
                    else:
                        print('偏转速度：', vel_turn)
                        self.control.set_motor(0, abs(vel_turn), math.copysign(0.5, (target_x - 300)))

                    prev_deflection = deflection
                    prev_target_y = target_y
                else:
                    if not self.grabbed:
                        print('前进抓球')                    
                        self.control.move(0, 0.4, 0.45) # 0.5s
                        self.control.set_servo1(130)
                        self.control.set_servo2(55)
                        self.control.move(0,0.35,-0.5) 
                        self.grabbed = True
                        self.control.move(-0.5, 0.5, 1.55) 
                        self.control.set_motor(0,0,0)
                        self.control.move(0,0.6,0.8)
                    break

    # def FindBalls(self):
    #     p1 = 0.0013  # 0.0013
    #     p2 = 0.00135  # 0.0012 0.00135
    #     p21 = 0.0015  # 0.0016
    #     d1 = 0.00025   #  0.0001
    #     d2 = 0.00027   #  0.0001
    #     prev_deflection = 0  
    #     prev_target_y = 0 

    #     while True:
    #         # ti=time.time()
    #         _, frame = self.cap.read()
    #         cls, self.pos = self.detector.detect_Balls(frame, self.index) # ,save_path=f'/home/orangepi/SmartSave/Saving_demo/PredResult/img{ti}.jpg'

    #         # PID找球
    #         if self.pos == (0, 0):
    #             self.control.set_motor(0, 0.3, -0.5)
    #             print('未发现')
    #         else:                           
    #             target_x, target_y = self.pos
    #             if target_y > 356: #366 362
    #                     while(abs(target_x - 300)>5):
    #                         print('近距离调中心')
    #                         _, frame = self.cap.read()
    #                         cls, self.pos = self.detector.detect_Balls(frame, self.index)
    #                         target_x, _ = self.pos
    #                         vel_turn = p21 * (target_x - 300)
    #                         print(f'中心坐标：{self.pos}')
    #                         print(f'偏转速度：{vel_turn}')
    #                         self.control.set_motor(0, abs(vel_turn), math.copysign(0.5, vel_turn))
    #                     print('张开爪子')
    #                     self.control.set_servo1(106) #90
    #                     self.control.set_servo2(74)
    #             if target_y < 382:  # 430 385
    #                 self.grabbed = False
    #                 vel_forward = p1 * abs(640 - target_y)
    #                 vel_turn = p2 * (target_x - 300)
    #                 deflection = target_x - 300

    #                 print(f'{cls}位置：', self.pos)

    #                 delta_deflection = deflection - prev_deflection
    #                 delta_target_y = target_y - prev_target_y

    #                 vel_forward += d1 * delta_target_y
    #                 vel_turn += d2 * delta_deflection

    #                 if abs(deflection) < 20 or abs(vel_turn) < 0.08: # or abs(vel_turn) < 0.1 0.15 0.07
    #                     print('前进速度：', vel_forward)
    #                     self.control.set_motor(0, vel_forward, 0)
    #                 else:
    #                     print('偏转速度：', vel_turn)
    #                     self.control.set_motor(0, abs(vel_turn), math.copysign(0.5, (target_x - 300)))

    #                 prev_deflection = deflection
    #                 prev_target_y = target_y
    #             else:
    #                 if not self.grabbed:
    #                     print('前进抓球')                    
    #                     self.control.move(0, 0.4, 0.45) # 0.5s
    #                     self.control.set_servo1(130)
    #                     self.control.set_servo2(55)
    #                     self.control.move(0,0.35,-0.6) 
    #                     self.grabbed = True
    #                     self.control.move(-0.5, 0.5, 1.8) 
    #                     self.control.set_motor(0,0,0)
    #                     self.control.move(0,0.7,0.8)
    #                 break

    def FindBalls(self):
        p1 = 0.0013  # 0.0013
        p2 = 0.0015  # 0.0012 0.00135
        p21 = 0.0015  # 0.0016
        d1 = 0.0002  #  0.0001 0.00025 
        d2 = 0.00025   #  0.0001 0.0002
        prev_deflection = 0  
        prev_target_y = 0 

        while True:
            # ti=time.time()
            _, frame = self.cap.read()
            cls, self.pos = self.detector.detect_Balls(frame, self.index) # ,save_path=f'/home/orangepi/SmartSave/Saving_demo/PredResult/img{ti}.jpg'

            # PID找球
            if self.pos == (0, 0):
                self.control.set_motor(0, 0.25, -0.5)
                print('未发现')
            else:                           
                target_x, target_y = self.pos
                if target_y > 365: #366 362
                        # while(abs(target_x - 300)>10):
                        #     print('近距离调中心')
                        #     _, frame = self.cap.read()
                        #     cls, self.pos = self.detector.detect_Balls(frame, self.index)
                        #     target_x, _ = self.pos
                        #     vel_turn = p21 * (target_x - 300)
                        #     print(f'中心坐标：{self.pos}')
                        #     print(f'偏转速度：{vel_turn}')
                        #     self.control.set_motor(0, abs(vel_turn), math.copysign(0.5, vel_turn))
                        print('张开爪子')
                        self.control.set_servo1(106) #90
                        self.control.set_servo2(74)
                if target_y < 386:  # 430 385
                    self.grabbed = False
                    vel_forward = p1 * abs(640 - target_y)
                    vel_turn = p2 * (target_x - 300)
                    deflection = target_x - 300

                    print(f'{cls}位置：', self.pos)

                    delta_deflection = deflection - prev_deflection
                    delta_target_y = target_y - prev_target_y

                    vel_forward += d1 * delta_target_y
                    vel_turn += d2 * delta_deflection

                    if abs(deflection) < 20 or abs(vel_turn) < 0.08: # or abs(vel_turn) < 0.1 0.15 0.07
                        print('前进速度：', vel_forward)
                        self.control.set_motor(0, vel_forward, 0)
                    else:
                        print('偏转速度：', vel_turn)
                        self.control.set_motor(0, abs(vel_turn), math.copysign(0.5, (target_x - 300)))

                    prev_deflection = deflection
                    prev_target_y = target_y
                else:
                    if not self.grabbed:
                        print('前进抓球')                    
                        self.control.move(0, 0.5, 0.45) # 0.5s
                        self.control.set_servo1(130)
                        self.control.set_servo2(55)
                        self.control.move(0,0.35,-0.5) 
                        self.grabbed = True
                        self.control.move(-0.5, 0.5, 1.55) 
                        self.control.set_motor(0,0,0)
                        self.control.move(0,0.6,0.8)
                    break

    def FindArea(self):
        p1 = 0.0045 # 0.0045
        p2 = 0.0015 # 0.0016 0.00155
        kp1 = 0.0017
        kp2 = 0.0014 # 0.0014
        p21 = 0.004
                          
        while True:
            _, frame = self.cap.read()
            cls, self.pos1 = self.detector1.detect_Area(frame, self.index) #,save_path=f'/home/orangepi/SmartSave/Saving_demo/PredResult/img{ti}.jpg'
            target_x, target_y = self.pos1

            datahex = self.ser.read(33)
            while (self.ser.in_waiting > 33):
                datahex = self.ser.read(33)
            angle = DueData(datahex)
            angle = angle - self.init_angle  
            if angle > 180:
                angle -= 360  
            elif angle < -180:
                angle += 360
            angle = -angle
            print(angle)

            if self.pos1 == (0, 0) :
                self.control.set_motor(0, 0.3, -0.5)  # 0.35 修改
            else:
                if cls == self.ares[self.index]:
                    print(f'{cls}位置：', self.pos1)
                    if 290>target_y > 280: #366 362
                        while(abs(target_x - 315)>3):
                            print('近距离调中心')
                            _, frame = self.cap.read()
                            _, self.pos1 = self.detector1.detect_Area(frame, self.index)
                            target_x, _ = self.pos1
                            vel_turn = p21 * (target_x - 300) if target_x>0 else 0
                            print(f'中心坐标：{self.pos1}')
                            print(f'偏转速度：{vel_turn}')
                            self.control.set_motor(0, abs(vel_turn), math.copysign(0.5, vel_turn))
                    if target_y < 300: #310
                        self.finded = False
                        vel_forward = (300 - target_y) * p1 if abs(300 - target_y) * p1 > 0.11 else 0.11
                        deflection = (target_x - 300)
                        vel_turn = deflection * p2 

                        if abs(deflection) < 25 : #or abs(vel_turn) < 0.09
                            self.control.set_motor(0, vel_forward, 0)
                            print('前进速度：', vel_forward)
                        else:
                            self.control.set_motor(0, abs(vel_turn), math.copysign(0.5, vel_turn))
                            print('偏转速度：', vel_turn)
                    else:
                        if not self.finded:
                            if 15 < angle < 70:
                                print('角度在(10,70)')
                                self.control.move(0, 0.5, -0.5)
                                self.IMU_PID(0)
                                self.control.move(0, 0.6, 0.5)
                                self.IMU_PID(-90)
                                self.control.move(0, 0.65, 0.55) #0.6s
                            elif 120 < angle < 170:
                                print('角度在(120,170)')
                                self.control.move(0, 0.5, -0.5)
                                self.IMU_PID(-180)
                                self.control.move(0, 0.6, 0.5)
                                self.IMU_PID(-90)
                                self.control.move(0, 0.65, 0.55) #0.6s
                            else:
                                self.control.move(0, 0.6, 0.5) # 0.5 0.6s
                                self.control.set_motor(0, 0, 0)

                            print(f'开始放球')
                            self.control.set_servo1(90)
                            self.control.set_servo2(90)
                            time.sleep(0.4)
                            self.control.move(0, 0.5, -1)
                            self.control.set_servo1(130)
                            self.control.set_servo2(55)
                            self.control.move(-0.5, 0.5, 1.2) # .move(-0.5, 0.6, 1.5)
                            self.finded = True
                            break
                        # else:
                        #     self.control.move(0, 0.5, -0.7)
                        #     self.control.move(-0.5, 0.5, 1.2) #.move(-0.5, 0.6, 1.5)
                        #     break 
                elif cls == "cross":
                    print(f'{cls}位置：', self.pos1)
                    if target_y < 360: #380 355
                        vel_forward = kp1 * abs(640 - target_y)
                        vel_turn = kp2 * (target_x - 300)
                        deflection = target_x - 300
                        if abs(deflection) < 30 or abs(vel_turn) < 0.15:
                            print('前进速度：', vel_forward)
                            self.control.set_motor(0, vel_forward, 0)
                        else:
                            print('偏转速度：', vel_turn)
                            self.control.set_motor(0, abs(vel_turn), math.copysign(0.5, vel_turn))   
                    else:
                        self.IMU_PID(-80) 

    def FindArea1(self):
        p1 = 0.0045 # 0.0045
        p2 = 0.00157 # 0.0016 0.00155
        kp1 = 0.0015
        kp2 = 0.0014 # 0.0014
                          
        while True:
            _, frame = self.cap.read()
            cls, self.pos1 = self.detector1.detect_Area(frame, self.index) #,save_path=f'/home/orangepi/SmartSave/Saving_demo/PredResult/img{ti}.jpg'
            target_x, target_y = self.pos1

            datahex = self.ser.read(33)
            while (self.ser.in_waiting > 33):
                datahex = self.ser.read(33)
            angle = DueData(datahex)
            angle = angle - self.init_angle  
            if angle > 180:
                angle -= 360  
            elif angle < -180:
                angle += 360
            angle = -angle

            if self.pos1 == (0, 0) : # or cls == "cross" 修改部分
                self.control.set_motor(0, 0.25, 0.5)  # 0.35 修改
            else:
                if cls == self.ares[self.index]: # 修改部分
                    print(f'{cls}位置：', self.pos1)
                    if target_y < 300: #310
                        self.finded = False
                        vel_forward = (300 - target_y) * p1 if abs(300 - target_y) * p1 > 0.11 else 0.11
                        deflection = (target_x - 300)
                        vel_turn = deflection * p2 

                        if abs(deflection) < 25 : #or abs(vel_turn) > 0.09 30
                            self.control.set_motor(0, vel_forward, 0)
                            print('前进速度：', vel_forward)
                        else:
                            self.control.set_motor(0, abs(vel_turn), math.copysign(0.5, vel_turn))
                            print('偏转速度：', vel_turn)
                    else:
                        if not self.finded:
                            if 0 < angle < 70:
                                print('角度在(0,70)')
                                self.control.move(0, 0.5, -0.5)
                                self.IMU_PID(0)
                                self.control.move(0, 0.6, 0.5)
                                self.IMU_PID(-90)
                                self.control.move(0, 0.5, 0.7) #0.6s
                            elif 110 < angle < 180:
                                print('角度在(110,180)')
                                self.control.move(0, 0.5, -0.5)
                                self.IMU_PID(-180)
                                self.control.move(0, 0.6, 0.5)
                                self.IMU_PID(-90)
                                self.control.move(0, 0.5, 0.7) #0.6s
                            else:
                                self.control.move(0, 0.5, 0.55) # 0.5 0.6s
                                self.control.set_motor(0, 0, 0)

                            print(f'开始放球')
                            self.control.set_servo1(90)
                            self.control.set_servo2(90)
                            time.sleep(0.4)
                            self.control.move(0, 0.7, -1.3)
                            self.control.set_servo1(130)
                            self.control.set_servo2(55)
                            time.sleep(0.3)
                            self.finded = True
                            break
                        else:
                            self.control.move(0, 0.7, -1.2)
                            break 
                elif cls == "cross":
                    print(f'{cls}位置：', self.pos1)
                    if target_y < 360: #380 355
                        vel_forward = kp1 * abs(640 - target_y)
                        vel_turn = kp2 * (target_x - 300)
                        deflection = target_x - 300
                        if abs(deflection) < 30 or abs(vel_turn) < 0.15:
                            print('前进速度：', vel_forward)
                            self.control.set_motor(0, vel_forward, 0)
                        else:
                            print('偏转速度：', vel_turn)
                            self.control.set_motor(0, abs(vel_turn), math.copysign(0.5, vel_turn))   
                    else:
                        self.IMU_PID(-80)                    
            
    def isFinded(self):
        while True:
            _, frame = self.cap.read()
            cls= self.detector.detect(frame , self.index)
            if cls=='blue_ball_in':
                print(f"Detected {cls},结束找第一个球")
                self.find_flag=True
                break
            else:
                print('继续找第一个球')
                self.find_flag=False
                break               
                               
    def run(self):
        self.detector=ObjectDetector()
        self.detector1=ObjectDetector1()

        self.control.buzzer(3)
        # self.control.set_servo(155)   
        self.control.set_servo1(130) 
        self.control.set_servo2(55)
        self.flag = False
        while True:
            # dat=self.control.button()
            # if dat:    
            #     if dat[-3]==1: 
            #         self.index=0 
            #         self.flag=True
            #         print(self.index)
            #     elif dat[-3]==2:
            #         self.index=1
            #         self.flag=True
            #         print(self.index)
                    
            # while self.flag:
                    # self.control.move(0, 1, 1.25)
                    # # self.control.set_motor(0,0,0)
                    # self.control.move(0, 0.75, -1.3)
                    # while not self.find_flag:
                    #     print('找第一个球')
                    #     self.first_d()
                    #     print("寻找安全区")
                    #     self.FindArea1()
                    #     # self.control.move(0, 0.4, -0.5)
                    #     self.isFinded()
                    # while True:
                        print('寻找球')      
                        self.FindBalls()
                        print("寻找安全区")
                        self.FindArea()
                        # self.IMU_PID(180)
                        break

    def runProcessing(self):
        run_process = Process(target=self.run)
        # IMU_process = Process(target=self.check_IMU)
        handleDetect_process = Process(target=self.check_handle)
        run_process.start()
        # IMU_process.start()
        handleDetect_process.start()
        handleDetect_process.join()
        # run_process.join()
        print('手动模式')
        run_process.kill()
        # IMU_process.kill()
        self.handle_process = Process(target=self.handler.start())
        self.handle_process.start()
        self.handle_process.join()
    
if __name__ == "__main__":
    car = CarRun()
    car.runProcessing()
    # car.run()

