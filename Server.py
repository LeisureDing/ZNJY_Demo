import numpy as np
import cv2
from socket import *

# 服务器：接收端 电脑接收数据/图片
s = socket(AF_INET, SOCK_DGRAM)
addr = ('192.168.236.77', 8080)  # 电脑连香橙派同一局域网的 IP
s.bind(addr)
print("Start listening on ...")

while True:
    # 接收数据时动态确定数据大小
    data, _ = s.recvfrom(65535)  # 增加缓冲区
    receive_data = np.frombuffer(data, dtype='uint8')

    r_img = cv2.imdecode(receive_data, 1)
    if r_img is None:
        print("Failed to decode image.")
        continue
    # cv2.putText(r_img, "server", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow('server_frame', r_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()