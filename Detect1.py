import time
import numpy as np
import cv2
from rknnlite.api import RKNNLite
from Camera1 import Camera1
from Client import *

class ObjectDetector1:
    def __init__(self):
        self.rknn = RKNNLite()
        self.model_path = '/home/orangepi/SmartSave/Saving_demo/weight/yolo5s_Area.rknn'
        self.obj_thresh = 0.80
        self.nms_thresh = 0.1
        self.img_size = 640
        self.save_path = '/home/orangepi/SmartSave/Saving_demo/PredResult/'
        self.classes = ("blue_start", "red_start", "blue_safe", "red_safe", "cross")
        self.Are = ["blue_safe", "red_safe"]
        self.tracking_class = None
        self.tracking_box = None

        self.load_model()
        self.init_runtime()

    def load_model(self):
        print('--> Load RKNN model')
        ret = self.rknn.load_rknn(self.model_path)
        if ret != 0:
            raise ValueError('Load RKNN model failed!')
        print('done')

    def init_runtime(self):
        ret = self.rknn.init_runtime()
        if ret != 0:
            raise ValueError('Init runtime environment failed!')
        print('done')

    def xywh2xyxy(self, x):
        y = np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2
        y[:, 1] = x[:, 1] - x[:, 3] / 2
        y[:, 2] = x[:, 0] + x[:, 2] / 2
        y[:, 3] = x[:, 1] + x[:, 3] / 2
        return y

    def process(self, input, mask, anchors):
        anchors = [anchors[i] for i in mask]
        grid_h, grid_w = map(int, input.shape[0:2])

        box_confidence = input[..., 4]
        box_confidence = np.expand_dims(box_confidence, axis=-1)

        box_class_probs = input[..., 5:]

        box_xy = input[..., :2] * 2 - 0.5
        col = np.tile(np.arange(0, grid_w), grid_w).reshape(-1, grid_w)
        row = np.tile(np.arange(0, grid_h).reshape(-1, 1), grid_h)
        col = col.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
        row = row.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
        grid = np.concatenate((col, row), axis=-1)
        box_xy += grid
        box_xy *= int(self.img_size / grid_h)

        box_wh = pow(input[..., 2:4] * 2, 2)
        box_wh = box_wh * anchors

        box = np.concatenate((box_xy, box_wh), axis=-1)

        return box, box_confidence, box_class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        boxes = boxes.reshape(-1, 4)
        box_confidences = box_confidences.reshape(-1)
        box_class_probs = box_class_probs.reshape(-1, box_class_probs.shape[-1])

        _box_pos = np.where(box_confidences >= self.obj_thresh)
        boxes = boxes[_box_pos]
        box_confidences = box_confidences[_box_pos]
        box_class_probs = box_class_probs[_box_pos]

        class_max_score = np.max(box_class_probs, axis=-1)
        classes = np.argmax(box_class_probs, axis=-1)
        _class_pos = np.where(class_max_score >= self.obj_thresh)

        boxes = boxes[_class_pos]
        classes = classes[_class_pos]
        scores = (class_max_score * box_confidences)[_class_pos]

        return boxes, classes, scores

    def nms_boxes(self, boxes, scores):
        x = boxes[:, 0]
        y = boxes[:, 1]
        w = boxes[:, 2] - boxes[:, 0]
        h = boxes[:, 3] - boxes[:, 1]

        areas = w * h
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x[i], x[order[1:]])
            yy1 = np.maximum(y[i], y[order[1:]])
            xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
            yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

            w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
            h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
            inter = w1 * h1

            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= self.nms_thresh)[0]
            order = order[inds + 1]
        keep = np.array(keep)
        return keep

    def yolov5_post_process(self, input_data):
        masks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        anchors = [[46, 115], [48, 104], [51, 117], [53, 107], [56, 116], [59, 99], [61, 124], [63, 109], [71, 100]]
        # 修改
        target_classes=("blue_safe", "red_safe", "cross")
        boxes, classes, scores = [], [], []
        for input, mask in zip(input_data, masks):
            b, c, s = self.process(input, mask, anchors)
            b, c, s = self.filter_boxes(b, c, s)
            boxes.append(b)
            classes.append(c)
            scores.append(s)

        boxes = np.concatenate(boxes)
        boxes = self.xywh2xyxy(boxes)
        classes = np.concatenate(classes)
        scores = np.concatenate(scores)

        nboxes, nclasses, nscores = [], [], []
        for c in set(classes):
            if self.classes[c] in target_classes:
                inds = np.where(classes == c)
                b = boxes[inds]
                c = classes[inds]
                s = scores[inds]

                keep = self.nms_boxes(b, s)

                nboxes.append(b[keep])
                nclasses.append(c[keep])
                nscores.append(s[keep])

        if not nclasses and not nscores:
            return None, None, None

        boxes = np.concatenate(nboxes)
        classes = np.concatenate(nclasses)
        scores = np.concatenate(nscores)

        return boxes, classes, scores

    def draw_results(self, frame, boxes, scores, classes, ratio, dw, dh):
        detected_objects = []  
        for box, score, cl in zip(boxes, scores, classes):
            box = box.copy()
            box[0] = int((box[0] - dw) / ratio[0])
            box[1] = int((box[1] - dh) / ratio[1])
            box[2] = int((box[2] - dw) / ratio[0])
            box[3] = int((box[3] - dh) / ratio[1])

            # 绘制检测框
            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)
            cv2.putText(frame, '{0} {1:.2f}'.format(self.classes[cl], score),
                        (int(box[0]), int(box[1] - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 255), 2)

            # 计算中心点并绘制十字标记
            center_x = int((box[0] + box[2]) / 2)
            center_y = int((box[1] + box[3]) / 2)
            cv2.drawMarker(frame, (center_x, center_y), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)

            detected_objects.append((self.classes[cl], (center_x, center_y)))

        # 返回绘制后的图像
        return frame, detected_objects

    def letterbox(self, im, new_shape=(640, 640), color=(0, 0, 0)):
        shape = im.shape[:2]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        ratio = r, r
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]

        dw /= 2
        dh /= 2

        if shape[::-1] != new_unpad:
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return im, ratio, (dw, dh)

    def save_output_image(self, frame):
        filename = f"{self.save_path}img.jpg"
        cv2.imwrite(filename, frame)

    def detect_Area(self, frame, save_path=None):
        img, ratio, (dw, dh) = self.letterbox(frame, new_shape=(self.img_size, self.img_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.expand_dims(img, 0)

        outputs = self.rknn.inference(inputs=[img])

        input0_data = outputs[0].reshape([3, -1] + list(outputs[0].shape[-2:]))
        input1_data = outputs[1].reshape([3, -1] + list(outputs[1].shape[-2:]))
        input2_data = outputs[2].reshape([3, -1] + list(outputs[2].shape[-2:]))

        input_data = [np.transpose(input0_data, (2, 3, 0, 1)),
                    np.transpose(input1_data, (2, 3, 0, 1)),
                    np.transpose(input2_data, (2, 3, 0, 1))]

        boxes, classes, scores = self.yolov5_post_process(input_data)

        if boxes is not None:
            processed_frame, detected_objects = self.draw_results(frame, boxes, scores, classes, ratio, dw, dh)
            for obj in detected_objects:
                return obj[0], obj[1], processed_frame 
        else:
            return None, (0, 0), frame 
        

if __name__ == '__main__':
    detector = ObjectDetector1()
    cap = Camera1(0)
    cap.init_cameras()
    
    while True:
        frame = cap.get_frame(0)
        t1 = time.time()
        cls, center ,img = detector.detect_Area(frame)
        handle_frames('192.168.106.77', 8080, img)
        t2 = time.time()
        print(f"Detected: Class: {cls}, Center: {center}")  
        print(f"Inference time: {((t2 - t1) * 1000):.2f}ms")