import time
import numpy as np
import cv2
from rknnlite.api import RKNNLite
from Camera1 import Camera1
from Client import *

class ObjectDetector:
    def __init__(self):
        self.model_path1 = '/home/orangepi/SmartSave/Saving_demo/weight/balls.rknn' #'/home/orangepi/SmartSave/Saving_demo/weight/yolo5s_Balls_quant.rknn' 
        
        # self.model_path1 = '/home/orangepi/SmartSave/Saving_demo/weight/ZHJY.rknn' # yolo5s_Balls_quant.rknn
        self.class_names1 = ("yellow_ball", "black_ball", "blue_ball", "red_ball", "yellow_ball_in", "black_ball_in", "blue_ball_in", "red_ball_in")
        self.obj_thresh = 0.72  # 0.80
        self.nms_thresh = 0.01  # 合理的NMS阈值
        self.img_size = 640

        self.rknn = RKNNLite()
        ret1 = self.rknn.load_rknn(self.model_path1)
        if ret1 != 0:
            print('Load RKNN model1 failed')
            exit(ret1)
        ret1 = self.rknn.init_runtime()
        if ret1 != 0:
            print('Init runtime 1 environment failed!')
            exit(ret1)
        self.Choice = ["blue_ball", "red_ball"]
        self.highScore = ["yellow_ball", "black_ball"]
        self.Find = ["blue_ball_in", "red_ball_in"]
        self._target_class_index = -1

    def __del__(self):
        self.rknn.release()

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

    def yolov5_process(self, input_data):
        masks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        anchors = [[46, 115], [48, 104], [51, 117], [53, 107], [56, 116], [59, 99],
                   [61, 124], [63, 109], [71, 100]]
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
            inds = np.where(classes == c)
            b = boxes[inds]
            c_cl = classes[inds]
            s = scores[inds]
            keep = self.nms_boxes(b, s)
            nboxes.append(b[keep])
            nclasses.append(c_cl[keep])
            nscores.append(s[keep])
        if not nclasses and not nscores:
            return None, None, None
        boxes = np.concatenate(nboxes)
        classes = np.concatenate(nclasses)
        scores = np.concatenate(nscores)
        return boxes, classes, scores
    
    # def detect_FirsBall(self, frame, index):
    #     t1 = time.time()
    #     img, ratio, (dw, dh) = self.letterbox(frame, new_shape=(self.img_size, self.img_size))
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #     img = np.expand_dims(img, 0)

    #     outputs = self.rknn.inference(inputs=[img])

    #     input_data = []
    #     for output in outputs:
    #         output = output.reshape((3, -1, output.shape[-2], output.shape[-1]))
    #         input_data.append(output.transpose((2, 3, 0, 1)))  # 合并操作
    #     boxes, classes, scores = self.yolov5_process(input_data)

    #     # 绘制检测框
    #     pro_img = cv2.cvtColor(img[0], cv2.COLOR_RGB2BGR).copy()
    #     detections = []

    #     if boxes is not None and classes is not None:
    #         # 获取当前要检测的类别
    #         target_class = self.Choice[index]
    #         if target_class not in self.class_names1:
    #             return None, (0, 0), None  # 返回 None 表示没有检测到
    #         target_class_index = self.class_names1.index(target_class)
    #         mask = (classes == target_class_index)
    #         current_boxes = boxes[mask]
    #         current_classes = classes[mask]
    #         current_scores = scores[mask]

    #         # 非极大值抑制
    #         keep = self.nms_boxes(current_boxes, current_scores)
    #         if keep.dtype not in [np.int32, np.int64, np.bool_]:
    #             keep = keep.astype(np.int64)  # 确保keep是整数类型

    #         current_boxes = current_boxes[keep]
    #         current_classes = current_classes[keep]
    #         current_scores = current_scores[keep]

    #         if len(current_boxes) > 0:
    #             # 根据 Y 坐标排序，选择最大的
    #             y_coords = current_boxes[:, 1]  
    #             max_y_idx = np.argmax(y_coords)
    #             selected_box = current_boxes[max_y_idx]
    #             selected_score = current_scores[max_y_idx]
    #             selected_class = current_classes[max_y_idx]

    #             x_min, y_min, x_max, y_max = map(int, selected_box)
    #             cv2.rectangle(pro_img, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
    #             cv2.putText(pro_img, f'{self.class_names1[selected_class]} {selected_score:.2f}', (x_min, y_min - 6),
    #                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    #             center_x = (x_min + x_max) // 2
    #             center_y = (y_min + y_max) // 2
    #             cv2.line(pro_img, (center_x - 5, center_y), (center_x + 5, center_y), (0, 255, 0), 2)
    #             cv2.line(pro_img, (center_x, center_y - 5), (center_x, center_y + 5), (0, 255, 0), 2)

    #             detections.append({
    #                 'class': self.class_names1[selected_class],
    #                 'center': (center_x, center_y)
    #             })
    #     t2 = time.time()
    #     print(f"Inference time: {((t2 - t1) * 1000):.2f}ms")

    #     if len(detections) > 0:
    #         return detections[0]['class'], detections[0]['center'], pro_img
    #     else:
    #         return None, (0, 0), frame

    def detect_FirsBall(self, frame, index):
        """
        稳定追踪：同类多目标时，用 IOU 与上一帧关联，避免坐标跳变
        返回: (class, center, vis_img)
        """
        import time
        t1 = time.time()

        # ---------- 1. 前处理 ----------
        img, ratio, (dw, dh) = self.letterbox(frame, new_shape=(self.img_size, self.img_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.expand_dims(img, 0)
        outputs = self.rknn.inference(inputs=[img])
        input_data = [out.reshape((3, -1, out.shape[-2], out.shape[-1])).transpose(2, 3, 0, 1)
                    for out in outputs]
        boxes, classes, scores = self.yolov5_process(input_data)

        vis_img = cv2.cvtColor(img[0], cv2.COLOR_RGB2BGR).copy()

        if boxes is None or len(boxes) == 0:
            self.last_box = None          # 无框重置
            return None, (0, 0), frame

        # ---------- 2. 类别过滤 ----------
        target_cls = self.Choice[index]
        tgt_idx = self.class_names1.index(target_cls)
        mask = classes == tgt_idx
        boxes = boxes[mask]
        scores = scores[mask]

        if len(boxes) == 0:
            self.last_box = None
            return None, (0, 0), vis_img

        # ---------- 3. IOU 帧间关联 ----------
        if not hasattr(self, 'last_box') or self.last_box is None:
            # 首帧或丢帧：直接选置信度最高
            best_idx = np.argmax(scores)
        else:
            # 计算与上一帧框的 IOU
            def iou(b1, b2):
                x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
                x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
                inter = max(0, x2 - x1) * max(0, y2 - y1)
                area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
                area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
                return inter / (area1 + area2 - inter + 1e-6)

            ious = [iou(b, self.last_box) for b in boxes]
            max_iou_idx = np.argmax(ious)
            if ious[max_iou_idx] >= 0.5:      # IOU 阈值可调
                best_idx = max_iou_idx
            else:
                best_idx = np.argmax(scores)   # 关联失败，重新选

        # ---------- 4. 更新并绘制 ----------
        x1, y1, x2, y2 = map(int, boxes[best_idx])
        self.last_box = boxes[best_idx]       # 保存供下一帧使用
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        # cv2.rectangle(vis_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        # cv2.putText(vis_img, f'{target_cls}:{scores[best_idx]:.2f}',
        #             (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        # cv2.drawMarker(vis_img, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, 10, 2)

        # print(f"Inference+track time: {((time.time() - t1) * 1000):.2f}ms")
        return target_cls, (cx, cy), vis_img
    
    def detect_HighBall(self, frame, index):
        """
        稳定追踪：同类多目标时，用 IOU 与上一帧关联，避免坐标跳变
        返回: (class, center, vis_img)
        """
        import time
        t1 = time.time()

        # ---------- 1. 前处理 ----------
        img, ratio, (dw, dh) = self.letterbox(frame, new_shape=(self.img_size, self.img_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.expand_dims(img, 0)
        outputs = self.rknn.inference(inputs=[img])
        input_data = [out.reshape((3, -1, out.shape[-2], out.shape[-1])).transpose(2, 3, 0, 1)
                    for out in outputs]
        boxes, classes, scores = self.yolov5_process(input_data)

        vis_img = cv2.cvtColor(img[0], cv2.COLOR_RGB2BGR).copy()

        if boxes is None or len(boxes) == 0:
            self.last_box = None          # 无框重置
            return None, (0, 0), frame

        # ---------- 2. 类别过滤 ----------
        target_cls = self.highScore[index]
        tgt_idx = self.class_names1.index(target_cls)
        mask = classes == tgt_idx
        boxes = boxes[mask]
        scores = scores[mask]

        if len(boxes) == 0:
            self.last_box = None
            return None, (0, 0), vis_img

        # ---------- 3. IOU 帧间关联 ----------
        if not hasattr(self, 'last_box') or self.last_box is None:
            # 首帧或丢帧：直接选置信度最高
            best_idx = np.argmax(scores)
        else:
            # 计算与上一帧框的 IOU
            def iou(b1, b2):
                x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
                x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
                inter = max(0, x2 - x1) * max(0, y2 - y1)
                area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
                area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
                return inter / (area1 + area2 - inter + 1e-6)

            ious = [iou(b, self.last_box) for b in boxes]
            max_iou_idx = np.argmax(ious)
            if ious[max_iou_idx] >= 0.5:      # IOU 阈值可调 0.3
                best_idx = max_iou_idx
            else:
                best_idx = np.argmax(scores)   # 关联失败，重新选

        x1, y1, x2, y2 = map(int, boxes[best_idx])
        self.last_box = boxes[best_idx]       # 保存供下一帧使用
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        # cv2.rectangle(vis_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        # cv2.putText(vis_img, f'{target_cls}:{scores[best_idx]:.2f}',
        #             (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        # cv2.drawMarker(vis_img, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, 10, 2)

        # print(f"Inference+track time: {((time.time() - t1) * 1000):.2f}ms")
        return target_cls, (cx, cy), vis_img

    def detect_Balls(self, frame, index, save_path=f'/home/orangepi/SmartSave/Saving_demo/PredResult/img.jpg'):
        t1 = time.time()
        priority_target_classes = ["yellow_ball", "black_ball", self.Choice[index]] # , self.Choice[index]
        img, ratio, (dw, dh) = self.letterbox(frame, new_shape=(self.img_size, self.img_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.expand_dims(img, 0)

        outputs = self.rknn.inference(inputs=[img])

        input_data = [output.reshape((3, -1, output.shape[-2], output.shape[-1])).transpose((2, 3, 0, 1))
                    for output in outputs]
        boxes, classes, scores = self.yolov5_process(input_data)

        pro_img = cv2.cvtColor(img[0], cv2.COLOR_RGB2BGR).copy()
        detections = []

        if boxes is not None:
            current_target_class = None
            if self._target_class_index != -1:
                current_target_class = priority_target_classes[self._target_class_index]

            filtered_indices = []
            if current_target_class:
                target_class_index = self.class_names1.index(current_target_class)
                filtered_indices = np.where(classes == target_class_index)[0]
            else:
                for class_name in priority_target_classes:
                    class_index = self.class_names1.index(class_name)
                    indices = np.where(classes == class_index)[0]
                    if len(indices) > 0:
                        filtered_indices = indices
                        current_target_class = class_name
                        self._target_class_index = priority_target_classes.index(class_name)
                        break

            if len(filtered_indices) > 0:
                selected_boxes = boxes[filtered_indices]
                selected_scores = scores[filtered_indices]
                selected_classes = classes[filtered_indices]

                # 先按 y 降序，再按 x 升序排序
                sorted_indices = np.lexsort(
                    (-selected_boxes[:, 1], selected_boxes[:, 0]))  # y降序，x升序
                best_idx = sorted_indices[0]

                selected_box = selected_boxes[best_idx]
                selected_score = selected_scores[best_idx]
                selected_class = selected_classes[best_idx]

                top, left, right, bottom = map(int, selected_box)
                center_x = (top + right) // 2
                center_y = (left + bottom) // 2
                
                # cv2.rectangle(pro_img, (top, left), (right, bottom), (255, 0, 0), 2)
                # cv2.putText(pro_img, f'{self.class_names1[selected_class]} {selected_score:.2f}', (top, left - 6),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                # cv2.line(pro_img, (center_x - 5, center_y), (center_x + 5, center_y), (0, 255, 0), 2)
                # cv2.line(pro_img, (center_x, center_y - 5), (center_x, center_y + 5), (0, 255, 0), 2)

                detections.append({
                    'class': self.class_names1[selected_class],
                    'center': (center_x, center_y)
                })
            else:
                self._target_class_index = -1

        t2 = time.time()
        print(f"Inference time: {((t2 - t1) * 1000):.2f}ms")

        if detections:
            return (detections[0]['class'], detections[0]['center'], pro_img)
        else:
            return (None, (0, 0), frame)

    def detect_yellow_black(self, frame, save_path=f'/home/orangepi/SmartSave/Saving_demo/PredResult/img.jpg'):
        t1 = time.time()
        priority_target_classes = ["yellow_ball", "black_ball"] 
        img, ratio, (dw, dh) = self.letterbox(frame, new_shape=(self.img_size, self.img_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.expand_dims(img, 0)

        outputs = self.rknn.inference(inputs=[img])

        input_data = [output.reshape((3, -1, output.shape[-2], output.shape[-1])).transpose((2, 3, 0, 1))
                    for output in outputs]
        boxes, classes, scores = self.yolov5_process(input_data)

        pro_img = cv2.cvtColor(img[0], cv2.COLOR_RGB2BGR).copy()
        detections = []

        if boxes is not None:
            current_target_class = None
            if self._target_class_index != -1:
                current_target_class = priority_target_classes[self._target_class_index]

            filtered_indices = []
            if current_target_class:
                target_class_index = self.class_names1.index(current_target_class)
                filtered_indices = np.where(classes == target_class_index)[0]
            else:
                for class_name in priority_target_classes:
                    class_index = self.class_names1.index(class_name)
                    indices = np.where(classes == class_index)[0]
                    if len(indices) > 0:
                        filtered_indices = indices
                        current_target_class = class_name
                        self._target_class_index = priority_target_classes.index(class_name)
                        break

            if len(filtered_indices) > 0:
                selected_boxes = boxes[filtered_indices]
                selected_scores = scores[filtered_indices]
                selected_classes = classes[filtered_indices]

                # 先按 y 降序，再按 x 升序排序
                sorted_indices = np.lexsort(
                    (-selected_boxes[:, 1], selected_boxes[:, 0]))  # y降序，x升序
                best_idx = sorted_indices[0]

                selected_box = selected_boxes[best_idx]
                selected_score = selected_scores[best_idx]
                selected_class = selected_classes[best_idx]

                top, left, right, bottom = map(int, selected_box)
                center_x = (top + right) // 2
                center_y = (left + bottom) // 2
                
                # cv2.rectangle(pro_img, (top, left), (right, bottom), (255, 0, 0), 2)
                # cv2.putText(pro_img, f'{self.class_names1[selected_class]} {selected_score:.2f}', (top, left - 6),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                # cv2.line(pro_img, (center_x - 5, center_y), (center_x + 5, center_y), (0, 255, 0), 2)
                # cv2.line(pro_img, (center_x, center_y - 5), (center_x, center_y + 5), (0, 255, 0), 2)

                detections.append({
                    'class': self.class_names1[selected_class],
                    'center': (center_x, center_y)
                })
            else:
                self._target_class_index = -1

        t2 = time.time()
        print(f"Inference time: {((t2 - t1) * 1000):.2f}ms")

        if detections:
            return (detections[0]['class'], detections[0]['center'], pro_img)
        else:
            return (None, (0, 0), frame)

    def detect(self, frame, index, save_path=f'/home/orangepi/SmartSave/Saving_demo/PredResult/img.jpg'): # , save_path=None
        img, ratio, (dw, dh) = self.letterbox(frame, new_shape=(self.img_size, self.img_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.expand_dims(img, 0)

        outputs = self.rknn.inference(inputs=[img])

        input_data = []
        for output in outputs:
            output = output.reshape((3, -1, output.shape[-2], output.shape[-1]))
            input_data.append(output.transpose((2, 3, 0, 1)))  # 合并操作
        boxes, classes, scores = self.yolov5_process(input_data)

        # 绘制检测框
        pro_img = cv2.cvtColor(img[0], cv2.COLOR_RGB2BGR).copy()
        detections = []

        if boxes is not None and classes is not None:
            # 获取当前要检测的类别
            target_class = self.Find[index]
            if target_class not in self.class_names1:
                return None, (0, 0)  
            target_class_index = self.class_names1.index(target_class)
            mask = (classes == target_class_index)
            current_boxes = boxes[mask]
            current_classes = classes[mask]
            current_scores = scores[mask]

            # 非极大值抑制
            keep = self.nms_boxes(current_boxes, current_scores)
            if keep.dtype not in [np.int32, np.int64, np.bool_]:
                keep = keep.astype(np.int64)  # 确保keep是整数类型

            current_boxes = current_boxes[keep]
            current_classes = current_classes[keep]
            current_scores = current_scores[keep]

            if len(current_boxes) > 0:
                # 根据 Y 坐标排序，选择最大的
                y_coords = current_boxes[:, 1]  
                max_y_idx = np.argmax(y_coords)
                selected_box = current_boxes[max_y_idx]
                selected_score = current_scores[max_y_idx]
                selected_class = current_classes[max_y_idx]

                x_min, y_min, x_max, y_max = map(int, selected_box)
                cv2.rectangle(pro_img, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
                cv2.putText(pro_img, f'{self.class_names1[selected_class]} {selected_score:.2f}', (x_min, y_min - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                center_x = (x_min + x_max) // 2
                center_y = (y_min + y_max) // 2
                cv2.line(pro_img, (center_x - 5, center_y), (center_x + 5, center_y), (0, 255, 0), 2)
                cv2.line(pro_img, (center_x, center_y - 5), (center_x, center_y + 5), (0, 255, 0), 2)

                detections.append({
                    'class': self.class_names1[selected_class],
                    'center': (center_x, center_y)
                })
        # if save_path:
        #     cv2.imwrite(save_path, pro_img)

        if len(detections) > 0:
            return detections[0]['class']
        else:
            return None
        
    def jude_fistball(self, frame,index):
        """
        ROI: y 480~544, x 200~399
        返回值: flag, counts_dict
            flag = 0  -> 出现黄色球
            flag = 1  -> 只有蓝球或无球
        counts_dict -> {"yellow_ball":n, "black_ball":n, "blue_ball":n, "red_ball":n}
        """
        ball = ["blue_ball", "red_ball"]
        roi_y_min, roi_y_max = 480, 545
        roi_x_min, roi_x_max = 200, 400

        # 1. 前处理 + 推理
        img, ratio, (dw, dh) = self.letterbox(frame, new_shape=(self.img_size, self.img_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.expand_dims(img, 0)
        outputs = self.rknn.inference(inputs=[img])
        input_data = [
            out.reshape((3, -1, out.shape[-2], out.shape[-1])).transpose(2, 3, 0, 1)
            for out in outputs
        ]
        boxes, classes, scores = self.yolov5_process(input_data)

        counts = {"yellow_ball": 0, "black_ball": 0, "blue_ball": 0, "red_ball": 0}

        if boxes is None:
            return 1, counts  # 无球 → 1

        # 2. 遍历框并统计 ROI 内各类球
        for box, cls_idx in zip(boxes, classes):
            x1, y1, x2, y2 = box
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            if roi_x_min <= cx < roi_x_max and roi_y_min <= cy < roi_y_max:
                cls_name = self.class_names1[int(cls_idx)]
                if cls_name in counts:
                    counts[cls_name] += 1

        # 3. 判断规则
        yellow_cnt = counts["yellow_ball"]
        blue_cnt   = counts[ball[index]]
        other_cnt  = sum(counts[k] for k in counts if k not in {"yellow_ball", ball[index]})

        if yellow_cnt > 0:
            return 0, counts  # 出现黄球 → 0
        if blue_cnt >= 1 and other_cnt == 0:  # if blue_cnt >= 1 and other_cnt == 0:
            return 1, counts  # 只有蓝球 → 1
        # if blue_cnt < 2 and other_cnt == 0:
        #     return 2, counts  # 只有一个蓝球 → 2
        if blue_cnt == 0 and other_cnt == 0:
            return 0, counts  # 无球 → 0
        return 0, counts      # 其余情况 → 0

    def jude_yellow(self, frame):
        """
        统计 ROI 内各球数量并给出 yellow_ball 共存标志
        ROI: y 480~544, x 200~399
        返回值: flag, counts_dict
            flag  = 0  -> yellow_ball 与其他类共存
            flag  = 1  -> yellow_ball 单独存在或不存在
            counts_dict -> {"yellow_ball":n, "black_ball":n, ...}
        """
        roi_y_min, roi_y_max = 480, 545
        roi_x_min, roi_x_max = 200, 400

        # 1. 完整检测
        img, ratio, (dw, dh) = self.letterbox(frame, new_shape=(self.img_size, self.img_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.expand_dims(img, 0)
        outputs = self.rknn.inference(inputs=[img])
        input_data = [out.reshape((3, -1, out.shape[-2], out.shape[-1])).transpose(2, 3, 0, 1)
                    for out in outputs]
        boxes, classes, scores = self.yolov5_process(input_data)

        counts = {"yellow_ball": 0, "black_ball": 0, "blue_ball": 0, "red_ball": 0}

        if boxes is None:
            return 1, counts

        # 2. 遍历所有框，筛选 ROI
        for box, cls_idx in zip(boxes, classes):
            x1, y1, x2, y2 = box
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            if roi_x_min <= cx < roi_x_max and roi_y_min <= cy < roi_y_max:
                cls_name = self.class_names1[int(cls_idx)]
                if cls_name in counts:
                    counts[cls_name] += 1

        # 3. 判断共存规则
        yellow_cnt = counts["yellow_ball"]
        other_cnt = sum(counts[k] for k in counts if k != "yellow_ball")

        if (yellow_cnt > 0 and other_cnt > 0) or (yellow_cnt == 0 and other_cnt == 0) or (yellow_cnt == 2):
            return 0, counts
        # if yellow_cnt == 0 and other_cnt < 2:
        #     return 2, counts
        return 1, counts
    
        
if __name__ == '__main__': 
    detector = ObjectDetector()
    cap = Camera1(0)
    cap.init_cameras()
    
    while True:
        frame = cap.get_frame(0)
        ti = int(time.time())
    
        # cls, center , img = detector.detect_Balls(frame,0) # , save_path=f'/home/orangepi/SmartSave/Saving_demo/PredResult/img{ti}.jpg'
        cnt, cls = detector.jude_fistball(frame,0)
        print(f'检测到球类别：{cls}, 共存标志：{cnt}')
        # handle_frames('192.168.236.77', 8080, img)
