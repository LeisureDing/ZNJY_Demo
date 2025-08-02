import serial
import time
import struct
from Check import *
import threading
from Config import *
from IMU import *


##########控制器##########
class Controller():
    """
    控制指令编码发送给下位机.

    .03
    """
    def __init__(self) -> None:
        portx1 = "/dev/ttyACM0"  # 下位机
        portx2 = '/dev/ttyUSB0'  # IMU
        # portx1 = "COM13"
        # portx2 = "COM3"
    
        bps = 115200
        self.serial=serial.Serial(portx1, int(bps), timeout=0.0005,parity=serial.PARITY_NONE, stopbits=1) # 下位机
        self.serial1=serial.Serial(portx2, int(bps), timeout=0.0005, parity=serial.PARITY_NONE, stopbits=1) # IMU

        threading.Thread(target=self.linten_imu).start()

        self.kl = 1
        self.kr = 1
        self.kf = 1
        self.imu_value = 0
        self.acc_value = 0
        self.init_angle = None

    def linten_imu(self):
        datahex = bytes()
        while True:
            if self.serial1.in_waiting > 33:
                datahex = self.serial1.read(33)
                angle, acc = DueData(datahex)
                if self.init_angle is None:
                    self.init_angle = angle
                    logger.info(f"self.init_angle: {self.init_angle}")
                self.imu_value = angle
                self.acc_value = acc
                # logger.info(f"angle:{self.imu_value},acc:{self.acc_value}")
            else:
                time.sleep(0.004) # 0.02

    def _read_data(self):
        # if self.serial is not None and self.serial.is_open:
        if self.serial and self.serial.is_open:
            result = self.serial.read(1024)  
            # hex_result = result.hex()  转16进制
            return result
        else:
            print("No detect")
            return None

    def _send_data(self, data):
        self.serial.write(data)
        time.sleep(0.02)

    # 控制2个电机速度
    def set_motor(self, mode:int,speed:float, angle:float) -> None:
        """
        控制 0和1号电机转速及转向
        AA 55 03 0C 01 02 00 00 00 80 BF 01 00 00 00 40 FB
        """
        front = (3, 3) # 2.8
        right = (3, -3)
        left = (-3, 3)  # angel<0
        back = (3,2.8)

        right1 = (6, 4) # (6, 3.95)
        left1 = (4, 6)
        angle_val=float(angle)

        front_weight = (self.kf * max(0, (1 - abs(angle_val))))
        right_weight = (self.kr * max(0, angle_val))
        left_weight = (self.kl * max(0, -angle_val))
        back_weight = (self.kf * max(0, (1 - abs(angle_val))))

        if mode==0:
            if angle_val == 0:
                speed1 = front[0] * front_weight * speed
                speed2 = front[1] * front_weight * -speed
            elif angle_val < 0:
                speed1 = left[0] * left_weight * speed
                speed2 = left[1] * left_weight * -speed
            else:
                speed1 = right[0] * right_weight * speed
                speed2 = right[1] * right_weight * -speed
        elif mode==1:
            if angle_val == 0:
                speed1 = front[0] * front_weight * speed
                speed2 = front[1] * front_weight * -speed
            elif angle_val < 0:
                speed1 = left1[0] * left_weight * speed
                speed2 = left1[1] * left_weight * -speed
            else:
                speed1 = right1[0] * right_weight * speed
                speed2 = right1[1] * right_weight * -speed
        else:
            speed1 = back[0] * back_weight * speed
            speed2 = back[1] * back_weight * -speed

        speed1_hex = struct.pack('f', speed1).hex().upper()
        speed2_hex = struct.pack('f', speed2).hex().upper()
        # print(speed1,speed2)
        check=checksum_crc8(bytes.fromhex('030C010201'+''.join(speed1_hex) +'02'+''.join(speed2_hex)))
        data = 'AA55030C0102 01'+''.join(speed1_hex)+'02'+''.join(speed2_hex) + ''.join(check)
        self._send_data(bytes.fromhex(data))

    # 控制2个电机停止
    def stop(self) -> None:
        """
        0和1号电机停止
        AA55030203 03 AD
        """
        data = 'AA55030203 03 70'
        self._send_data(bytes.fromhex(data))

   # 控制1号PWM舵机（摄像头）
    def set_servo(self, angle: int) -> None:
        """
        控制PWM舵机 0 运动到 1000 位置  脉冲宽度为[ 500 , 2500 ],对应的是[ 0°,180°]
        AA55 0406 03 时间:e803(1s)  id:00  pulse:e8 03 e4
        """
        pulse=angle_to_pulse(angle)
        pulse_hex= "{:04X}".format(pulse)
        pulse_=pulse_hex[2:4]+pulse_hex[0:2]
        check=checksum_crc8(bytes.fromhex('0406 03 0100 01' + ''.join(pulse_)))
        data = 'AA55 0406 03 0100 01' + ''.join(pulse_) + ''.join(check)
        self._send_data(bytes.fromhex(data))

    # 控制2，3号PWM舵机（框）
    def set_servo1(self, angle: int) -> None:
        """
        控制PWM舵机 0 运动到 1000 位置  脉冲宽度为[ 500 , 2500 ],对应的是[ 0°,180°]
        AA55 0406 03 时间:e803(1s)  id:00  pulse:e8 03 e4
        """
        pulse=angle_to_pulse(angle)
        pulse_hex= "{:04X}".format(pulse)
        pulse_=pulse_hex[2:4]+pulse_hex[0:2]
        check=checksum_crc8(bytes.fromhex('0406 03 0100 04' + ''.join(pulse_)))
        data = 'AA55 0406 03 0100 04' + ''.join(pulse_) + ''.join(check)
        self._send_data(bytes.fromhex(data))

    def buzzer(self,times:int) -> None:
        """
        蜂鸣器
        AA 55 02 08 78 05 64 00 64 00 循环次数5:05 00 F0
        """
        times_hex = "{:04X}".format(times)
        times_=times_hex[2:4]+times_hex[0:2]
        check=checksum_crc8(bytes.fromhex('0208 7805 6400 6400' + ''.join(times_)))
        data = 'AA55 0208 7805 6400 6400'+''.join(times_)+ ''.join(check)
        self._send_data(bytes.fromhex(data))

    def led(self)->None:
        """
        LED闪烁
        """
        data='AA 55 01 07 01 64 00 64 00 05 00 37'
        self._send_data(bytes.fromhex(data))

    # def button(self):
    #     if self.serial.inWaiting() > 0:
    #         data = self.serial.read(self.serial.inWaiting())
    #         # print(data)
    #         if data:
    #             # print(data[-3])
    #             return data[-3]
    #     # return None

    def button(self):
        if self.serial.inWaiting() > 0:  # 只要有数据就读取
            data = self.serial.read(self.serial.inWaiting())
            if data:
                # print(data)  # 打印所有接收到的数据
                return data

    def move(self,angle: float,run_time:float, speed: int) -> None:
        """
        :angle:车角度
        :run_time: 运行时间
        :speed: 速度
        """
        if angle ==0:
            if speed>0:
                self.set_motor(0,speed,0)
                time.sleep(run_time)  
                self.set_motor(0,0,0)
            elif speed<0:
                self.set_motor(2,speed,0)
                time.sleep(run_time)
                self.set_motor(0,0,0)  
        elif angle>0: # 右转
            self.set_motor(0,speed,angle)
            time.sleep(run_time)
            self.set_motor(0,0,0)
        else : # 左转
            self.set_motor(0,speed,angle)
            time.sleep(run_time)
            self.set_motor(0,0,0)

    def backword(self,run_time:float, speed: int) -> None:
        self.set_motor(2,speed,0)
        time.sleep(run_time)  
        self.set_motor(0,0,0)

if __name__ == '__main__':
    car = Controller()
    car.buzzer(3)
    # car.set_servo(40) # 抬起摄像头高度40   90
    # car.set_servo1(75)  # 接近时->148 = 136 ; 抬起-> 90 = 75 ; 框下-> 155 = 145; 
    cnt=0  # 以下面速度转3圈是800
    # car.move(0, 0.4, -2)

    # car.move(0, 1, -1) 
    # car.set_motor(0, 0.36, 0.5) 
    # time.sleep(1)
    car.set_motor(0, 0, 0)
    # while True:
    #     car.set_motor(0, 0.26, -0.5)
    #     cnt+=1
    #     print(f"CNT={cnt}")

    # 第一个球
    # car.set_servo1(100) 
    # car.set_servo2(80)
    # car.move(0, 0.78, 1.5)
    # car.set_motor(0, 0, 0)
    # car.set_servo1(135) 
    # car.move(1,0.4,0.4)
    # car.move(0, 0.4, 1)
    # car.set_motor(0, 0, 0)

    # car.set_motor(1,0,0)
    # car.set_servo1(110) 
    # car.set_servo2(70)
    # while True:  
        # dat=car.button()
        # if dat:    
        #     if dat[-3]==1: 
        #         index=0 
        #         print(index)
        #     elif dat[-3]==2:
        #         index=1
        #         print(index)

    # while True:
    #     while car.button():
    #         car.move(0,1,0.5)
