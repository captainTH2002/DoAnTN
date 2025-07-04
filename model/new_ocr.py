import onnxruntime as ort
import pdb
import json
import cv2
import os
import numpy as np
import pyclipper
from shapely.geometry import Polygon
import math
from copy import deepcopy
from PIL import Image
from pathlib import Path
import omegaconf


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
    cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 255)
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
        rect = np.zeros((4, 2), dtype = "float32")
        s = pts.sum(axis = 1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis = 1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect


    def four_point_transform(self, image, pts):
        rect = self.order_points(pts)
        (tl, tr, br, bl) = rect
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        
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


class OCR:
    def __init__(self, common_cfg, model_cfg):
        self.common_cfg = common_cfg
        self.model_cfg = model_cfg
        self.session = ort.InferenceSession(self.model_cfg['model_path'], providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape[2:]
        self.max_sequence_length = self.model_cfg['max_sequence_length']
        charset = self.model_cfg['charset']
        self.charset_list = ['[E]'] + list(tuple(charset))

    def resize(self, im):
        height, width = self.input_shape[:2]
        h, w, d = im.shape
        unpad_im = cv2.resize(im, (int(height*w/h), height), interpolation=cv2.INTER_AREA)
        if unpad_im.shape[1] > width:
            im = cv2.resize(im, (width, height), interpolation=cv2.INTER_AREA)
        else:
            im = cv2.copyMakeBorder(unpad_im, 0, 0, 0, width-int(height*w/h), cv2.BORDER_CONSTANT, value=[0, 0, 0])
        return im

    def decode(self, p):
        cands = []
        for cand in p:
            if np.argmax(cand) == 0:
                break
            cands.append(cand)
        return cands

    def index_to_word(self, output):
        res = ''
        for probs in output:
            if np.argmax(probs) == 0:
                break
            else:
                res += self.charset_list[np.argmax(probs)]
        return res

    def predict(self, result):
        batch_images = []
        for j, image in enumerate(result['text_detection']['boxes_image']):
            resized_image = self.resize(image)
            processed_image = Image.fromarray(resized_image).convert('RGB')
            processed_image = np.array([np.transpose(np.array(processed_image)/255., (2, 0, 1))]).astype(np.float32)
            normalized_image = (processed_image - 0.5) / 0.5
            batch_images.append(normalized_image[0])

        batch_images_length = len(batch_images)
        while len(batch_images) % self.model_cfg['max_batch_size'] != 0:
            batch_images.append(batch_images[0])

        batch_images = np.array(batch_images)
        text_output = []
        if len(batch_images) != 0:
            index = 0
            while index < len(batch_images):
                text_output += self.session.run(None, {self.input_name: batch_images[index:index+self.model_cfg['max_batch_size']]})
                index += self.model_cfg['max_batch_size']

        if len(text_output) > 0:
            text_output = np.concatenate(text_output, axis=0)
        else:
            text_output = np.array([])  # Hoặc giá trị mặc định khác

        text_output = text_output[:batch_images_length]  # shape (num_boxes, max_sequence_length, num_classes)
        raw_words = [self.index_to_word(prob) for prob in text_output]

        assert len(raw_words) == len(result['text_detection']['boxes_image'])
        result['ocr'] = {}
        result['ocr']['raw_words'] = raw_words
        return result


def poly2bb(poly):
    xmin, xmax = min(poly[::2]), max(poly[::2])
    ymin, ymax = min(poly[1::2]), max(poly[1::2])
    return (xmin, ymin, xmax, ymax)


def max_left(poly):
    return min(poly[0], poly[2], poly[4], poly[6])

def max_right(poly):
    return max(poly[0], poly[2], poly[4], poly[6])

def row_polys(polys):
    polys.sort(key=lambda x: max_left(x))
    clusters, y_min = [], []
    for tgt_node in polys:
        if len (clusters) == 0:
            clusters.append([tgt_node])
            y_min.append(tgt_node[1])
            continue
        matched = None
        tgt_7_1 = tgt_node[7] - tgt_node[1]
        min_tgt_0_6 = min(tgt_node[0], tgt_node[6])
        max_tgt_2_4 = max(tgt_node[2], tgt_node[4])
        max_left_tgt = max_left(tgt_node)
        for idx, clt in enumerate(clusters):
            src_node = clt[-1]
            src_5_3 = src_node[5] - src_node[3]
            max_src_2_4 = max(src_node[2], src_node[4])
            min_src_0_6 = min(src_node[0], src_node[6])
            overlap_y = (src_5_3 + tgt_7_1) - (max(src_node[5], tgt_node[7]) - min(src_node[3], tgt_node[1]))
            overlap_x = (max_src_2_4 - min_src_0_6) + (max_tgt_2_4 - min_tgt_0_6) - (max(max_src_2_4, max_tgt_2_4) - min(min_src_0_6, min_tgt_0_6))
            if overlap_y > 0.5*min(src_5_3, tgt_7_1) and overlap_x < 0.6*min(max_src_2_4 - min_src_0_6, max_tgt_2_4 - min_tgt_0_6):
                distance = max_left_tgt - max_right(src_node)
                if matched is None or distance < matched[1]:
                    matched = (idx, distance)
        if matched is None:
            clusters.append([tgt_node])
            y_min.append(tgt_node[1])
        else:
            idx = matched[0]
            clusters[idx].append(tgt_node)
    
    # Sửa lỗi inhomogeneous shape
    zip_clusters = list(zip(clusters, y_min))
    zip_clusters.sort(key=lambda x: x[1])
    
    # Trích xuất chỉ clusters từ zip_clusters đã sắp xếp
    sorted_clusters = [cluster for cluster, _ in zip_clusters]
    
    return sorted_clusters


def sort_polys(polys):
    poly_clusters = row_polys(polys)
    polys = []
    for row in poly_clusters:
        polys.extend(row)
    return polys, poly_clusters


def compute_boxes_iou(box1, box2):
    x1, y1, x2, y2 = box1
    poly1 = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
    x1, y1, x2, y2 = box2
    poly2 = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
    intersect = poly1.intersection(poly2).area
    union = poly1.union(poly2).area
    iou = intersect / union
    max_overlap_ratio = intersect / min(poly1.area, poly2.area)
    return iou, max_overlap_ratio

def verbalizer_layout(sorted_bbs, mapper):
    '''
    Return text with additional spaces character to reconstruct layout
    '''

    # Get min character height and width:
    heights, char_widths = [], []
    texts = []
    for i, line in enumerate(sorted_bbs):
        for j, bb in enumerate(line): 
            bb_tuple = tuple(bb)
            if mapper[bb_tuple]['text'] == '':
                continue
            x1, y1, x2, y2 = min(bb[::2]), min(bb[1::2]), max(bb[::2]), max(bb[1::2])
            mapper[bb_tuple]['rect_box'] = [x1, y1, x2, y2]
            mapper[bb_tuple]['line'] = i
            texts.append(mapper[bb_tuple])
            height, width = y2-y1, x2-x1
            char_width = width // len(mapper[bb_tuple]['text'])
            heights.append(height)
            char_widths.append(char_width)
    min_char_height = min(heights)
    min_char_width = min(char_widths)
    raw_text = texts[0]['text']
    for i in range(1, len(texts)):
        if texts[i]['line'] == texts[i-1]['line']: #Same line as previous text
            # Caculate number of space must be inserted
            width_distance = texts[i]['rect_box'][0] - texts[i-1]['rect_box'][2]
            num_space = int(max(1, width_distance // min_char_width))
            raw_text += ' '*num_space
        else: #New line
            raw_text += '\n'
        raw_text += texts[i]['text']
    return raw_text

class PostProcessor:
    def __init__(self, common_cfg, model_cfg) -> None:
        self.common_cfg = common_cfg
        self.model_cfg = model_cfg

    def predict(self, result):
        """
            format result from ocr and text detection and layout detection into paddle format
        """
        src_img = result['orig_img']

        # get region texts
        poly2text = {}
        for poly, text in zip(result['text_detection']['coords'], result['ocr']['raw_words']):
            poly2text[poly] = {'text':text}

        polys = result['text_detection']['coords']
        poly_rows = row_polys(polys)
        # texts = verbalizer_layout(poly_rows, poly2text)
        # return texts
        texts = []
        for row in poly_rows:
            for poly in row:
                texts.append(poly2text[poly]['text'])
            texts.append('\n')
        
        return ' '.join(texts)
    
    
class Processor:
    def __init__(self, common_cfg, model_cfg):
        self.common_cfg = common_cfg
        self.model_cfg = model_cfg
        self.modules = [
            TextDetector(common_cfg, model_cfg['text_detection']),
            OCR(common_cfg, model_cfg['ocr']),

        ]

    def predict(self, image):
        result = {'orig_img': image}
        for module in self.modules:
            result = module.predict(result)
        return result

def process_images_in_folder(folder_path, processor):
    # Duyệt tất cả ảnh trong thư mục
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # Lọc các file ảnh
            img_path = os.path.join(folder_path, filename)
            print(f"Processing: {img_path}")
            
            img = cv2.imread(img_path)
            if img is None:
                print(f"Error loading {img_path}")
                continue  # Bỏ qua nếu ảnh không load được

            # Chạy dự đoán trên ảnh
            result = processor.predict(img)

            # Trích xuất thông tin OCR và tọa độ
            work = result['ocr']['raw_words']
            coords = result['text_detection']['coords']
            size = img.shape[:2]

            # Tạo file JSON cho từng ảnh
            json_fp = os.path.join(folder_path, f"{os.path.splitext(filename)[0]}.json")
            to_json(work, coords, size, img_path, json_fp)

# ... (Giữ nguyên toàn bộ code trước đó, chỉ thay đổi phần main)

def main(image_input):
    # Cấu hình mô hình
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_cfg = {
        'text_detection': {
            'model_path': os.path.join(base_dir, 'models/text_detection/epoch=43_val_total_loss=0.66.onnx')
        },
        'ocr': {
            'model_path': os.path.join(base_dir, 'models/ocr/onnx_print_v4_batch8.onnx'),
            'max_sequence_length': 31,
            'charset': "&1#@'()+,-./0123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZ_`abcdefghijklmnopqrstuvwxyz<>ÀÁÂÃÇÈÉÊÌÍÒÓÔÕÙÚÝàáâãçèéêìíòóôõùúýĂăĐđĨĩŨũƠơƯưẠạẢảẤấẦầẨẩẪẫẬậẮắẰằẲẳẴẵẶặẸẹẺẻẼẽẾếỀềỂểỄễỆệỈỉỊịỌọỎỏỐốỒồỔổỖỗỘộỚớỜờỞởỠỡỢợỤụỦủỨứỪừỬửỮữỰựỲỳỴỵỶỷỸỹ'"";* $%",
            'max_batch_size': 8,
        },
        'postprocess': {}
    }
    common_cfg = {}

    # Khởi tạo Processor
    processor = Processor(common_cfg, model_cfg)

    # Kiểm tra đầu vào là file ảnh hay thư mục
    input_path = Path(image_input)
    if input_path.is_file() and input_path.suffix.lower() in ('.png', '.jpg', '.jpeg'):
        # Xử lý một ảnh đơn lẻ
        img = cv2.imread(str(input_path))
        if img is None:
            print(f"Error loading {input_path}")
            return
        
        result = processor.predict(img)
        work = result['ocr']['raw_words']
        coords = result['text_detection']['coords']
        size = img.shape[:2]
        json_fp = input_path.with_suffix('.json')
        to_json(work, coords, size, str(input_path), str(json_fp))
        print(f"Processed {input_path}, JSON saved at {json_fp}")
    
    elif input_path.is_dir():
        # Xử lý tất cả ảnh trong thư mục
        for filename in os.listdir(input_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = input_path / filename
                img = cv2.imread(str(img_path))
                if img is None:
                    print(f"Error loading {img_path}")
                    continue
                
                result = processor.predict(img)
                work = result['ocr']['raw_words']
                coords = result['text_detection']['coords']
                size = img.shape[:2]
                json_fp = img_path.with_suffix('.json')
                to_json(work, coords, size, str(img_path), str(json_fp))
                print(f"Processed {img_path}, JSON saved at {json_fp}")
    else:
        print(f"Invalid input: {image_input}. Must be an image file or directory.")

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python script.py <image_path_or_directory>")
        sys.exit(1)
    main(sys.argv[1])
