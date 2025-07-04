import cv2
import numpy as np
import onnxruntime as ort
import os
import json
import base64
import pyclipper
from shapely.geometry import Polygon
import math
from copy import deepcopy
from PIL import Image
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from functools import lru_cache

# ===== TextDetector =====
class TextDetector:
    def __init__(self, model_path):
        # Sử dụng tất cả các luồng CPU có sẵn bằng cách đặt số luồng thành 0
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 0
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # Sử dụng ExecutionProvider có sẵn hiệu suất tốt nhất
        providers = ort.get_available_providers()
        provider_options = {}
        
        # Ưu tiên sử dụng CUDA nếu có
        if 'CUDAExecutionProvider' in providers:
            provider_options = [
                ('CUDAExecutionProvider', {
                    'device_id': 0,
                    'arena_extend_strategy': 'kNextPowerOfTwo',
                    'gpu_mem_limit': 2 * 1024 * 1024 * 1024,
                    'cudnn_conv_algo_search': 'EXHAUSTIVE',
                    'do_copy_in_default_stream': True,
                }),
                ('CPUExecutionProvider', {
                    'use_arena': True,
                })
            ]
            self.session = ort.InferenceSession(model_path, sess_options, providers=provider_options)
        else:
            self.session = ort.InferenceSession(model_path, sess_options, providers=['CPUExecutionProvider'])
        
        self.input_name = self.session.get_inputs()[0].name

    def order_points(self, pts):
        # Sắp xếp các điểm theo thứ tự: trên-trái, trên-phải, dưới-phải, dưới-trái
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
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
            [0, maxHeight - 1]], dtype="float32")
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

    def resize_image(self, image, image_short_side):
        h, w = image.shape[:2]
        if h < w:
            h_new = image_short_side
            w_new = int(w / h * h_new / 32) * 32
        else:
            w_new = image_short_side
            h_new = int(h / w * w_new / 32) * 32
        resized_img = cv2.resize(image, (w_new, h_new))
        return resized_img

    def preprocess(self, image):
        h, w = image.shape[:2]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.resize_image(image, image_short_side=640)
        image_input = np.expand_dims(image, axis=0).transpose(0, 3, 1, 2)
        return image_input

    def predict(self, src_image):
        h, w = src_image.shape[:2]
        tensor = self.preprocess(src_image)
        p = self.session.run(None, {self.input_name: tensor})[0][0]
        p = np.array(p).transpose(1, 2, 0) 
        bitmap = p > 0.3
        bbs, scores = self.polygons_from_bitmap(p, bitmap, w, h, box_thresh=0.3, max_candidates=-1)

        new_bbs = []
        bbs_raw = bbs
        for bb in bbs:
            x1, y1 = bb[0]
            x2, y2 = bb[1]
            x3, y3 = bb[2]
            x4, y4 = bb[3]
            new_bbs.append((x1, y1, x2, y2, x3, y3, x4, y4))

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
        result = {
            'coords': new_bbs,
            'boxes_image': boxes_image
        }
        return result

    def polygons_from_bitmap(self, pred, bitmap, dest_width, dest_height, max_candidates=100, box_thresh=0.7):
        height, width = bitmap.shape[:2]
        boxes, scores = [], []

        contours, _ = cv2.findContours((bitmap * 255.0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            epsilon = 0.002 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            points = approx.reshape((-1, 2))
            if points.shape[0] < 4:
                continue
            score = self.box_score_fast(pred, points.reshape((-1, 2)))
            if box_thresh > score:
                continue
            
            if points.shape[0] > 2:
                box = self.unclip(points, unclip_ratio=1.3)
                if len(box) > 1:
                    continue
            else:
                continue
            box = np.array(box).reshape(-1, 2)
            if len(box) == 0: continue
            box, sside = self.get_mini_boxes(box.reshape((-1, 1, 2)))
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

    def box_score_fast(self, bitmap, _box):
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

    def unclip(self, box, unclip_ratio=1.5):
        poly = Polygon(box)
        subject = [tuple(l) for l in box]
        distance = poly.area * unclip_ratio / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(subject, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = offset.Execute(distance)
        return expanded

    def get_mini_boxes(self, contour):
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

# ===== OCR =====
class OCR:
    def __init__(self, model_path, charset, max_sequence_length, max_batch_size=8):
        # Tối ưu session options
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 0
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.enable_profiling = False
        
        # Tối ưu providers
        providers = ort.get_available_providers()
        if 'CUDAExecutionProvider' in providers:
            provider_options = [
                ('CUDAExecutionProvider', {
                    'device_id': 0,
                    'arena_extend_strategy': 'kNextPowerOfTwo',
                    'gpu_mem_limit': 2 * 1024 * 1024 * 1024,
                    'cudnn_conv_algo_search': 'EXHAUSTIVE',
                    'do_copy_in_default_stream': True,
                }),
                ('CPUExecutionProvider', {
                    'use_arena': True,
                })
            ]
            self.session = ort.InferenceSession(model_path, sess_options, providers=provider_options)
        else:
            self.session = ort.InferenceSession(model_path, sess_options, providers=['CPUExecutionProvider'])
            
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape[2:]
        self.max_sequence_length = max_sequence_length
        self.charset_list = ['[E]'] + list(tuple(charset))
        # Tăng batch size cho hiệu suất tốt hơn nếu có nhiều vùng văn bản
        self.max_batch_size = max(max_batch_size, 16)  # Tối ưu batch size

    @lru_cache(maxsize=128)  # Cache kết quả của hàm resize để tránh tính toán lại
    def resize(self, im_shape, interp=cv2.INTER_AREA):
        """Tính toán kích thước mới và padding (nếu cần) cho ảnh với kích thước im_shape"""
        height, width = self.input_shape[:2]
        h, w, d = im_shape
        
        # Tính toán kích thước mới giữ nguyên tỷ lệ
        new_w = int(height * w / h)
        
        if new_w > width:
            # Nếu chiều rộng mới vượt quá width, thì resize trực tiếp
            return (width, height), None
        else:
            # Nếu cần padding
            return (new_w, height), width - new_w
    
    def resize_image(self, im):
        """Resize và pad ảnh thực tế"""
        h, w, d = im.shape
        (new_w, new_h), padding = self.resize((h, w, d))
        
        # Resize ảnh
        if padding is None:
            # Resize trực tiếp nếu không cần padding
            return cv2.resize(im, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        else:
            # Resize giữ nguyên tỷ lệ rồi pad
            unpad_im = cv2.resize(im, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            return cv2.copyMakeBorder(unpad_im, 0, 0, 0, padding, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    def index_to_word(self, output):
        res = ''
        for probs in output:
            if np.argmax(probs) == 0:
                break
            else:
                res += self.charset_list[np.argmax(probs)]
        return res

    def preprocess_image(self, image):
        """Tiền xử lý một ảnh để chuẩn bị cho mô hình OCR"""
        resized_image = self.resize_image(image)
        # Sử dụng trực tiếp OpenCV thay vì chuyển qua PIL để tăng tốc
        processed_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        # Chuẩn bị tensor
        processed_image = np.transpose(processed_image / 255.0, (2, 0, 1)).astype(np.float32)
        # Chuẩn hóa
        normalized_image = (processed_image - 0.5) / 0.5
        return normalized_image
    
    def batch_process_images(self, images, batch_size):
        """Xử lý song song các batch ảnh sử dụng đa luồng"""
        with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as executor:
            futures = [executor.submit(self.preprocess_image, img) for img in images]
            preprocessed_images = [future.result() for future in as_completed(futures)]
        
        # Pad batch nếu cần
        batch_images_length = len(preprocessed_images)
        pad_size = (batch_size - (batch_images_length % batch_size)) % batch_size
        if pad_size > 0:
            preprocessed_images.extend([preprocessed_images[0]] * pad_size)
        
        return np.array(preprocessed_images), batch_images_length
        
    def predict(self, boxes_image):
        """Dự đoán văn bản từ các ảnh đã cắt từ vùng văn bản"""
        if not boxes_image:
            return []
        
        # Tiền xử lý các ảnh song song
        preprocessed_batch, original_length = self.batch_process_images(boxes_image, self.max_batch_size)
        
        # Chạy inference
        text_output = []
        for i in range(0, len(preprocessed_batch), self.max_batch_size):
            batch = preprocessed_batch[i:i+self.max_batch_size]
            outputs = self.session.run(None, {self.input_name: batch})
            text_output.extend(outputs[0])  # Chỉ lấy outputs[0] vì đây là kết quả chính
        
        # Cắt kết quả về đúng kích thước ban đầu
        text_output = text_output[:original_length]
        
        # Chuyển đổi output thành văn bản
        raw_words = [self.index_to_word(prob) for prob in text_output]
        
        return raw_words

# ===== PostProcessor =====
class PostProcessor:
    def __init__(self):
        pass

    def max_left(self, poly):
        return min(poly[0], poly[2], poly[4], poly[6])

    def max_right(self, poly):
        return max(poly[0], poly[2], poly[4], poly[6])

    def row_polys(self, polys):
        polys.sort(key=lambda x: self.max_left(x))
        clusters, y_min = [], []
        for tgt_node in polys:
            if len(clusters) == 0:
                clusters.append([tgt_node])
                y_min.append(tgt_node[1])
                continue
            matched = None
            tgt_7_1 = tgt_node[7] - tgt_node[1]
            min_tgt_0_6 = min(tgt_node[0], tgt_node[6])
            max_tgt_2_4 = max(tgt_node[2], tgt_node[4])
            max_left_tgt = self.max_left(tgt_node)
            for idx, clt in enumerate(clusters):
                src_node = clt[-1]
                src_5_3 = src_node[5] - src_node[3]
                max_src_2_4 = max(src_node[2], src_node[4])
                min_src_0_6 = min(src_node[0], src_node[6])
                overlap_y = (src_5_3 + tgt_7_1) - (max(src_node[5], tgt_node[7]) - min(src_node[3], tgt_node[1]))
                overlap_x = (max_src_2_4 - min_src_0_6) + (max_tgt_2_4 - min_tgt_0_6) - (max(max_src_2_4, max_tgt_2_4) - min(min_src_0_6, min_tgt_0_6))
                if overlap_y > 0.5*min(src_5_3, tgt_7_1) and overlap_x < 0.6*min(max_src_2_4 - min_src_0_6, max_tgt_2_4 - min_tgt_0_6):
                    distance = max_left_tgt - self.max_right(src_node)
                    if matched is None or distance < matched[1]:
                        matched = (idx, distance)
            if matched is None:
                clusters.append([tgt_node])
                y_min.append(tgt_node[1])
            else:
                idx = matched[0]
                clusters[idx].append(tgt_node)
        zip_clusters = list(zip(clusters, y_min))
        zip_clusters.sort(key=lambda x: x[1])
        zip_clusters = list(np.array(zip_clusters, dtype=object)[:, 0])
        return zip_clusters

    def process(self, coords, raw_words):
        poly2text = {}
        for poly, text in zip(coords, raw_words):
            poly2text[poly] = {'text': text}

        poly_rows = self.row_polys(coords)
        texts = []
        for row in poly_rows:
            for poly in row:
                texts.append(poly2text[poly]['text'])
            texts.append('\n')
        
        return ' '.join(texts)

# ===== Utility Function =====
def encode_image_to_base64(image_path):
    with open(image_path, 'rb') as img_file:
        encoded = base64.b64encode(img_file.read()).decode('utf-8')
    return encoded

def to_json(work, coords, size, image_path, json_path):
    h, w = size
    json_dicts = {
        'shapes': [],
        'imagePath': image_path,
        'imageData': encode_image_to_base64(image_path),
        'imageHeight': h,
        'imageWidth': w
    }
    
    if len(coords) != len(work):
        raise ValueError("Number of coordinates must match the number of characters in work.")

    for char, coord in zip(work, coords):
        if len(coord) % 2 != 0:
            raise ValueError("Coordinate length must be even.")

        points = [[float(coord[i]), float(coord[i + 1])] for i in range(0, len(coord), 2)]
        json_dicts['shapes'].append({
            'text': char,
            'label': 'text',
            'points': points,
            'shape_type': 'polygon',
            'flags': {}
        })

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_dicts, f, indent=4, ensure_ascii=False)

# ===== Main Processor =====
class OCRProcessor:
    def __init__(self, text_detector_model_path, ocr_model_path, charset):
        self.text_detector = TextDetector(text_detector_model_path)
        self.ocr = OCR(
            ocr_model_path, 
            charset, 
            max_sequence_length=31, 
            max_batch_size=8
        )
        self.post_processor = PostProcessor()

    def process_image(self, image_path, output_json=True):
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error loading {image_path}")
            return None
        
        # Phát hiện vùng văn bản
        text_detection_result = self.text_detector.predict(img)
        
        # OCR các vùng văn bản
        raw_words = self.ocr.predict(text_detection_result['boxes_image'])
        
        # Hậu xử lý kết quả
        text_result = self.post_processor.process(text_detection_result['coords'], raw_words)
        
        result = {
            'text': text_result,
            'raw_words': raw_words,
            'coords': text_detection_result['coords']
        }
        
        # Lưu kết quả ra file JSON nếu cần
        if output_json:
            json_path = os.path.splitext(image_path)[0] + '.json'
            to_json(raw_words, text_detection_result['coords'], img.shape[:2], image_path, json_path)
        
        return result

    def process_folder(self, folder_path, output_json=True):
        results = {}
        
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(folder_path, filename)
                print(f"Processing: {img_path}")
                
                result = self.process_image(img_path, output_json)
                if result:
                    results[filename] = result
        
        return results

# ===== Main Function =====
def main():
    import argparse
    
    # Cấu hình tham số dòng lệnh
    parser = argparse.ArgumentParser(description='OCR cho hình ảnh hoặc thư mục chứa hình ảnh')
    parser.add_argument('input_path', type=str, help='Đường dẫn đến hình ảnh hoặc thư mục chứa hình ảnh')
    parser.add_argument('--text_detector', type=str, default='./models/text_detection/epoch=43_val_total_loss=0.66.onnx', 
                        help='Đường dẫn đến mô hình phát hiện văn bản')
    parser.add_argument('--ocr', type=str, default='./models/ocr/onnx_print_v4_batch8.onnx', 
                        help='Đường dẫn đến mô hình OCR')
    parser.add_argument('--output_json', action='store_true', default=True, 
                        help='Xuất kết quả ra file JSON')
    parser.add_argument('--verbose', action='store_true', 
                        help='Hiển thị thông tin chi tiết trong quá trình xử lý')
    
    args = parser.parse_args()
    
    # Charset định nghĩa tất cả các ký tự có thể nhận dạng
    charset = "&1#@'()+,-./0123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZ_`abcdefghijklmnopqrstuvwxyz<>ÀÁÂÃÇÈÉÊÌÍÒÓÔÕÙÚÝàáâãçèéêìíòóôõùúýĂăĐđĨĩŨũƠơƯưẠạẢảẤấẦầẨẩẪẫẬậẮắẰằẲẳẴẵẶặẸẹẺẻẼẽẾếỀềỂểỄễỆệỈỉỊịỌọỎỏỐốỒồỔổỖỗỘộỚớỜờỞởỠỡỢợỤụỦủỨứỪừỬửỮữỰựỲỳỴỵỶỷỸỹ'"";* $%"
    
    # Khởi tạo bộ xử lý OCR
    processor = OCRProcessor(args.text_detector, args.ocr, charset)
    
    input_path = args.input_path
    
    if os.path.isfile(input_path):
        # Xử lý một file ảnh
        if input_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(f"Đang xử lý ảnh: {input_path}")
            result = processor.process_image(input_path, args.output_json)
            if result:
                print(f"Kết quả OCR:")
                print(result['text'])
                print(f"Đã lưu JSON tại: {os.path.splitext(input_path)[0] + '.json'}" if args.output_json else "")
            else:
                print(f"Lỗi khi xử lý ảnh: {input_path}")
        else:
            print(f"Định dạng file không được hỗ trợ: {input_path}")
    
    elif os.path.isdir(input_path):
        # Xử lý tất cả ảnh trong thư mục
        print(f"Đang xử lý thư mục: {input_path}")
        results = processor.process_folder(input_path, args.output_json)
        
        # In kết quả tổng quan
        print(f"\nXử lý hoàn tất. Đã xử lý {len(results)} ảnh.")
        
        # In chi tiết nếu verbose
        if args.verbose and results:
            print("\nKết quả chi tiết:")
            for filename, result in results.items():
                print(f"\n--- {filename} ---")
                print(result['text'])
    
    else:
        print(f"Đường dẫn không tồn tại: {input_path}")

if __name__ == '__main__':
    main()