import cv2
import threading
import time
from Client import *

class Camera1:
    def __init__(self,main_id):
        self.camera_ids = {
            main_id: "主摄像头",
        }
        self.cameras = {}
        self.frames = {}
        self.update_flags = {}
        self.lock = threading.Lock()

    def init_cameras(self):
        for cam_id in self.camera_ids:
            cap = cv2.VideoCapture(cam_id)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

            for i in range(50):
                ret, frame = cap.read()
                if ret:
                    self.cameras[cam_id] = cap
                    self.frames[cam_id] = frame
                    self.update_flags[cam_id] = True
                    print(f"{self.camera_ids[cam_id]} 初始化成功")
                    break
                time.sleep(0.1)
            else:
                print(f"{self.camera_ids[cam_id]} 初始化失败")
                cap.release()
                self.release_all()
                exit(1)

        threading.Thread(target=self.update_frames, daemon=True).start()

    def update_frames(self):
        while True:
            for cam_id, cap in self.cameras.items():
                if self.update_flags.get(cam_id, False):
                    ret, frame = cap.read()
                    if ret:
                        with self.lock:
                            self.frames[cam_id] = frame

    def get_frame(self, cam_id):
        if cam_id in self.frames:
            with self.lock:
                return self.frames[cam_id].copy()
        else:
            print(f"无效的摄像头 ID:{cam_id}")
            return None

    def release_all(self):
        for cap in self.cameras.values():
            cap.release()
        print("所有摄像头已释放")
        
        
if __name__ == "__main__":
    cam = Camera1(0)
    cam.init_cameras()

    while True:
        frame = cam.get_frame(0)
        handle_frames('172.20.10.4' , 8080, frame)

