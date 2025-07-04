import onnxruntime as ort
import pdb
import json
import cv2
import numpy as np
import pyclipper
from shapely.geometry import Polygon
import math
from copy import deepcopy


def resize_image(image, image_short_side):
    h, w = image.shape[:2]
    if h < w:
        h_new = image_short_side
        w_new = int(w / h * h_new / 32) * 32
    else:
        w_new = image_short_side
        h_new = int(h / w * w_new / 32) * 32
    resized_img = cv2.resize(image, (w_new, h_new))
    return resized_img    

def box_score_fast(bitmap, _box):
    h, w = bitmap.shape[:2]
    box = _box.copy()
    xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int32), 0, w - 1)
    xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int32), 0, w - 1)
    ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int32), 0, h - 1)
    ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int32), 0, h - 1)

    mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
    box[:, 0] = box[:, 0] - xmin
    box[:, 1] = box[:, 1] - ymin
    cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
    return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]


def unclip(box, unclip_ratio=1.5):
    poly = Polygon(box)
    subject = [tuple(l) for l in box]
    distance = poly.area * unclip_ratio / poly.length
    offset = pyclipper.PyclipperOffset()
    offset.AddPath(subject, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    expanded = offset.Execute(distance)
    return expanded


def get_mini_boxes(contour):
    bounding_box = cv2.minAreaRect(contour)
    points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

    index_1, index_2, index_3, index_4 = 0, 1, 2, 3
    if points[1][1] > points[0][1]:
        index_1 = 0
        index_4 = 1
    else:
        index_1 = 1
        index_4 = 0
    if points[3][1] > points[2][1]:
        index_2 = 2
        index_3 = 3
    else:
        index_2 = 3
        index_3 = 2

    box = [points[index_1], points[index_2],
           points[index_3], points[index_4]]
    return box, min(bounding_box[1])


def polygons_from_bitmap(pred, bitmap, dest_width, dest_height, max_candidates=100, box_thresh=0.7):
    height, width = bitmap.shape[:2]
    boxes, scores = [], []

    contours, _ = cv2.findContours((bitmap * 255.0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        epsilon =  0.002* cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        points = approx.reshape((-1, 2))
        if points.shape[0] < 4:
            continue
        score = box_score_fast(pred, points.reshape((-1, 2)))
        if box_thresh > score:
            continue
        
        if points.shape[0] > 2:
            box = unclip(points, unclip_ratio=1.3)
            if len(box) > 1:
                continue
        else:
            continue
        box = np.array(box).reshape(-1, 2)
        if len(box) == 0: continue
        box, sside = get_mini_boxes(box.reshape((-1, 1, 2)))
        if sside < 5:
            continue
        box = np.array(box)
        box[:, 0] = np.clip(box[:, 0] / width * dest_width, 0, dest_width)
        box[:, 1] = np.clip(box[:, 1] / height * dest_height, 0, dest_height)
        boxes.append(box.astype('int32'))
        scores.append(score)
    if max_candidates == -1:
        return boxes, scores
    idxs = np.argsort(scores)
    scores = [scores[i] for i in idxs[:max_candidates]]
    boxes = [boxes[i] for i in idxs[:max_candidates]]
        
    return boxes, scores

def expand_boxes(boxes):
    new_boxes = []
    for box in boxes:
        box = np.array(box)
        box_h = max(box[3][1] - box[0][1], box[2][1] - box[1][1])
        box_w = max(box[1][0] - box[0][0], box[2][0] - box[3][0])
        if box_w / box_h < 4.0: 
            new_boxes.append(box)
            continue
        box[:2, 1] -= int(box_h * 10/100)
        box[2:, 1] += int(box_h * 10/100)
        new_boxes.append(box)
    return new_boxes

def prob2heatmap(prob_map, rgb=True):
    heatmap = cv2.applyColorMap((prob_map*255).astype('uint8'), cv2.COLORMAP_JET)
    if not rgb:
        heatmap = cv2.cvtColor(heatmap,cv2.COLOR_BGR2RGB)
    return heatmap

import base64

def encode_image_to_base64(image_path):
    with open(image_path, 'rb') as img_file:
        encoded = base64.b64encode(img_file.read()).decode('utf-8')
    return encoded


def to_json(work, coords, size, image_path, json_path, labels=None):
    h, w = size  # Lấy chiều cao và chiều rộng từ size
    json_dicts = {
        'shapes': [],
        'imagePath': image_path,
        'imageData': encode_image_to_base64(image_path),
        'imageHeight': h,
        'imageWidth': w
    }
    
    # Kiểm tra số lượng coords có tương ứng với số lượng ký tự trong work không
    if len(coords) != len(work):
        raise ValueError("Number of coordinates must match the number of characters in work.")

    # Duyệt qua từng ký tự và tọa độ tương ứng
    for char, coord in zip(work, coords):
        if len(coord) % 2 != 0:
            raise ValueError("Coordinate length must be even.")

        points = [[float(coord[i]), float(coord[i + 1])] for i in range(0, len(coord), 2)]
        if labels is None:
            json_dicts['shapes'].append({
                'text': char,  # Gán ký tự vào trường 'text'
                'label': 'text',
                'points': points,
                'shape_type': 'polygon',
                'flags': {}
            })
        else:
            # Kiểm tra số lượng labels có tương ứng với số lượng coords không
            if len(labels) != len(coords):
                raise ValueError("Number of labels must match number of coordinates.")
            
            for coord, label in zip(coords, labels):
                json_dicts['shapes'].append({
                    'label': label,
                    'points': points,
                    'shape_type': 'polygon',
                    'flags': {}
                })


    # Ghi dữ liệu ra file JSON
    with open(json_path, 'w', encoding='utf-8') as f:  # Đảm bảo mở file với encoding utf-8
        json.dump(json_dicts, f, indent=4, ensure_ascii=False)  # Thêm indent để dễ đọc file JSON
class TextDetector:
    def __init__(self, common_cfg, model_cfg):
        self.common_cfg = common_cfg
        self.model_cfg = model_cfg
        self.session = ort.InferenceSession(self.model_cfg['model_path'], providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name

    def order_points(self, pts):
        # initialzie a list of coordinates that will be ordered
        # such that the first entry in the list is the top-left,
        # the second entry is the top-right, the third is the
        # bottom-right, and the fourth is the bottom-left
        rect = np.zeros((4, 2), dtype = "float32")
        # the top-left point will have the smallest sum, whereas
        # the bottom-right point will have the largest sum
        s = pts.sum(axis = 1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        # now, compute the difference between the points, the
        # top-right point will have the smallest difference,
        # whereas the bottom-left will have the largest difference
        diff = np.diff(pts, axis = 1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        # return the ordered coordinates
        return rect


    def four_point_transform(self, image, pts):
        # obtain a consistent order of the points and unpack them
        # individually
        rect = self.order_points(pts)
        (tl, tr, br, bl) = rect
        # compute the width of the new image, which will be the
        # maximum distance between bottom-right and bottom-left
        # x-coordiates or the top-right and top-left x-coordinates
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        # compute the height of the new image, which will be the
        # maximum distance between the top-right and bottom-right
        # y-coordinates or the top-left and bottom-left y-coordinates
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        # now that we have the dimensions of the new image, construct
        # the set of destination points to obtain a "birds eye view",
        # (i.e. top-down view) of the image, again specifying points
        # in the top-left, top-right, bottom-right, and bottom-left
        # order
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype = "float32")
        # compute the perspective transform matrix and then apply it
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        return warped
    
    
    def expand_long_box(self, block, x1, y1, x2, y2, x3, y3, x4, y4):
        h, w, _ = block.shape
        box_h = max(y4 - y1, y3 - y2)
        box_w = max(x2 - x1, x3 - x4)
        if box_w / box_h >= 6:
            expand_pxt = math.ceil(0.1 * box_h)
            x1 = max(0, x1 - expand_pxt)
            x2 = min(w, x2 + expand_pxt)
            x3 = min(w, x3 + expand_pxt)
            x4 = max(0, x4 - expand_pxt)
            y1 = max(0, y1 - expand_pxt)
            y2 = max(0, y2 - expand_pxt)
            y3 = min(h, y3 + expand_pxt)
            y4 = min(h, y4 + expand_pxt)
        pts = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
        return pts
    
    
    def get_edge(self, x1, y1, x2, y2, x3, y3, x4, y4):
        e1 = math.sqrt(abs(x2 - x1)**2 + abs(y2 - y1)**2)
        e2 = math.sqrt(abs(x3 - x2)**2 + abs(y3 - y2)**2)
        e3 = math.sqrt(abs(x4 - x3)**2 + abs(y4 - y3)**2)
        e4 = math.sqrt(abs(x1 - x4)**2 + abs(y1 - y4)**2)
        edge_s = min([e1, e2, e3, e4])
        edge_l = max([e1, e2, e3, e4])
        return edge_s, edge_l
    
    
    def to_2_points(self, image, x1, y1, x2, y2, x3, y3, x4, y4):
        xmin = min(x1, x2, x3, x4)
        xmax = max(x1, x2, x3, x4)
        ymin = min(y1, y2, y3, y4)
        ymax = max(y1, y2, y3, y4)
        field_image = image[ymin:ymax, xmin:xmax]
        return field_image
    

    def preprocess(self, image):
        h, w= image.shape[:2]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = resize_image(image, image_short_side=640)
        image_input = np.expand_dims(image, axis=0).transpose(0, 3, 1, 2)#.astype('float32') #1, 3, x, 640
        return image_input


    def predict(self, result):
        src_image = deepcopy(result['orig_img'])
        h, w = src_image.shape[:2]
        tensor = self.preprocess(src_image)
        p = self.session.run(None, {self.input_name: tensor})[0][0]
        p = np.array(p).transpose(1, 2, 0) 
        bitmap = p > 0.3
        bbs, scores = polygons_from_bitmap(p, bitmap, w, h, box_thresh=0.3, max_candidates=-1)

        # Chuyển sang tuple thay vì danh sách
        new_bbs = []
        bbs_raw = bbs
        for bb in bbs:
            x1, y1 = bb[0]
            x2, y2 = bb[1]
            x3, y3 = bb[2]
            x4, y4 = bb[3]
            new_bbs.append((x1, y1, x2, y2, x3, y3, x4, y4))  # Sử dụng tuple

        boxes_image = []
        for box in bbs_raw:
            x1, y1 = box[0]
            x2, y2 = box[1]
            x3, y3 = box[2]
            x4, y4 = box[3]
            pts = self.expand_long_box(src_image, x1, y1, x2, y2, x3, y3, x4, y4)
            edge_s, edge_l = self.get_edge(x1, y1, x2, y2, x3, y3, x4, y4)
            if edge_l / edge_s < 1.5:
                text_image = self.to_2_points(src_image, x1, y1, x2, y2, x3, y3, x4, y4)
            else:
                text_image = self.four_point_transform(src_image, pts)
            boxes_image.append(text_image)

        assert len(new_bbs) == len(boxes_image)
        result['text_detection'] = {}
        result['text_detection']['coords'] = new_bbs  # new_bbs là tuple
        result['text_detection']['boxes_image'] = boxes_image

        

        return result
