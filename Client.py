import cv2
from socket import socket, AF_INET, SOCK_DGRAM
import threading

def send_img(img, addr):
    s = socket(AF_INET, SOCK_DGRAM)
    _, send_data = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 50])
    s.sendto(send_data.tobytes(), addr)
    # print(f'已发送{len(send_data)} Bytes的数据')
    s.close()

def process_and_send_frame(img, addr):
    th = threading.Thread(target=send_img, args=(img, addr))
    th.setDaemon(True)
    th.start()
    # img = cv2.flip(img, 0)  # 镜像翻转

def handle_frames(ip, port, frame):
    addr = (ip, port)
    if frame is None:
        print("接收到无效图像帧")
        return
    process_and_send_frame(frame, addr)
    cv2.waitKey(1)
    
# def frame_generator():
#     cap = cv2.VideoCapture(0) 
#     while True:
#         ret, img = cap.read()
#         if not ret:
#             break
#         return img
#     cap.release()
    
if __name__ == '__main__':
    handle_frames('192.168.219.77', 8080, img)