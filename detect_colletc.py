import pygame
import threading
import time
from Controller import Controller
import os
import cv2

pygame.init()

class HandleController:
    def __init__(self):
        self.joystick = None
        self.frame_count = 0
        if pygame.joystick.get_count() > 0:
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
            print(f"Connect {self.joystick.get_name()} Successfully!")
        else:
            print("No joystick detected.")

        self.controller = Controller()
        self.controller.set_motor(0, 0, 0)
        self.speed = 1.5
        self.speed_k = 0.8
        self.angle = 0
        self.angle_k = 0.4 # 0.6
        self.speed_lock = False
        self.running = False
        self.stop = False
        self.collect_threading = None
        self.camera = cv2.VideoCapture(0)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 640) 
        # 采集数据
        self.collect_flag = False
        self.collect_pause = False
        self.collect_now_n = 0
        self.collect_rounds = 0
        self.collect_last_n = 0
        self.collect_save_path = "Data"
        self.collect_threading: threading.Thread = None
        self.collect_lock = threading.Lock() 

    def start(self):
        self.running = True
        self.listen_handle()

    def listen_handle(self):
        s_t = time.time()
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.JOYBUTTONDOWN:
                    if event.button == 6:  # 左肩 张开
                        self.control.set_servo1(100) # 100
                        self.control.set_servo2(80)  # 80
                    elif event.button == 7:  # 右肩 闭合
                        self.controller.set_servo1(135)
                        self.controller.set_servo2(50)
                    # 停止 B
                    elif event.button == 1:
                        self.controller.set_motor(0,0,0)
                        print('Stop')

                    # 开始采集 A
                    elif event.button == 0:
                        if self.collect_flag:
                            if self.collect_pause:
                                print(f"第{self.collect_rounds}轮采集继续")
                                self.collect_pause = False
                            else:
                                print(f"第{self.collect_rounds}轮采集暂停")
                                self.collect_pause = True
                        else:
                            self.collect_flag = True
                            self.collect_pause = False
                            self.collect_threading = threading.Thread(target=self.collect)
                            self.collect_threading.start()
                            print(f"第{self.collect_rounds}轮采集开始")
                            
                    # 保存数据 X
                    elif event.button == 3:
                        if self.collect_pause:
                            if not os.path.exists(f"{self.collect_save_path}/{self.collect_rounds}/img/"):
                                os.makedirs(f"{self.collect_save_path}/{self.collect_rounds}/img/")
                            print(f"保存第{self.collect_rounds}轮采集数据,共采集{self.collect_now_n - self.collect_last_n}张图片")
                            self.collect_flag = False
                            self.collect_pause = False
                            self.collect_threading.join()
                            self.collect_threading = None
                            os.system(f"mv {self.collect_save_path}/tmp/* {self.collect_save_path}/{self.collect_rounds}/img/")
                            self.collect_last_n = self.collect_now_n
                            self.collect_rounds += 1

                    # 删除数据 Y
                    elif event.button == 4:
                        if self.collect_pause:
                            print(f"删除第{self.collect_rounds}轮采集数据")
                            self.collect_flag = False
                            self.collect_pause = False
                            self.collect_threading.join()
                            self.collect_threading = None
                            os.system(f"rm -rf {self.collect_save_path}/tmp/")
                            self.collect_now_n = self.collect_last_n

            # 左右方向
            anix0_value = self.joystick.get_axis(0) if abs(self.joystick.get_axis(0)) > 0.1 else 0
            # 前后方向
            anix3_value = self.joystick.get_axis(3) if abs(self.joystick.get_axis(3)) > 0.1 else 0

            # 计算速度和转向
            speed = -anix3_value * self.speed_k
            angle = anix0_value * self.angle_k

            # 如果同时检测到前后和左右方向输入，则同时设置速度和转向
            if abs(anix3_value) > 0.12 or abs(anix0_value) > 0.12:
                self.controller.set_motor(0,speed, angle)
                print(f"速度: {speed}, 转向: {angle}")
            else:
                self.controller.set_motor(0,0, 0)

        pygame.quit()

    def collect(self):
        if not os.path.exists(f"{self.collect_save_path}/tmp"):
            os.makedirs(f"{self.collect_save_path}/tmp")
        while self.collect_flag:
            if self.collect_pause:
                time.sleep(0.1)
                continue
            with self.collect_lock:  # 使用锁来确保线程安全
                # 获取图像
                _,frame = self.camera.read()
                cv2.imwrite(f"{self.collect_save_path}/tmp/{self.collect_now_n}.jpg", frame)
                self.collect_now_n += 1
            time.sleep(0.15)

if __name__ == "__main__":
    handleCollect = HandleController()
    handleCollect.start()