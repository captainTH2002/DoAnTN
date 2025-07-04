# import cv2
# import omegaconf
# import os  # Thêm thư viện os
# from text_detector import TextDetector, to_json
# from ocr import OCR
# from postprocessor import PostProcessor

# class Processor:
#     def __init__(self, common_cfg, model_cfg):
#         self.common_cfg = common_cfg
#         self.model_cfg = model_cfg
#         self.modules = [
#             TextDetector(common_cfg, model_cfg['text_detection']),
#             OCR(common_cfg, model_cfg['ocr']),
#             # PostProcessor(common_cfg, model_cfg['postprocess'])
#         ]

#     def predict(self, image):
#         result = {'orig_img': image}
#         for module in self.modules:
#             result = module.predict(result)
#         return result

# def process_images_in_folder(folder_path, processor):
#     # Duyệt tất cả ảnh trong thư mục
#     for filename in os.listdir(folder_path):
#         if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # Lọc các file ảnh
#             img_path = os.path.join(folder_path, filename)
#             print(f"Processing: {img_path}")
            
#             img = cv2.imread(img_path)
#             if img is None:
#                 print(f"Error loading {img_path}")
#                 continue  # Bỏ qua nếu ảnh không load được

#             # Chạy dự đoán trên ảnh
#             result = processor.predict(img)

#             # Trích xuất thông tin OCR và tọa độ
#             work = result['ocr']['raw_words']
#             coords = result['text_detection']['coords']
#             size = img.shape[:2]

#             # Tạo file JSON cho từng ảnh
#             json_fp = os.path.join(folder_path, f"{os.path.splitext(filename)[0]}.json")
#             to_json(work, coords, size, img_path, json_fp)

# def main():
#     # Cấu hình mô hình
#     model_cfg = {
#         'text_detection': {
#             'model_path': 'models/text_detection/epoch=43_val_total_loss=0.66.onnx'
#         },
#         'ocr': {
#             'model_path': 'models/ocr/onnx_print_v4_batch8.onnx',
#             'max_sequence_length': 31,
#             'charset': "&1#@'()+,-./0123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZ_`abcdefghijklmnopqrstuvwxyz<>ÀÁÂÃÇÈÉÊÌÍÒÓÔÕÙÚÝàáâãçèéêìíòóôõùúýĂăĐđĨĩŨũƠơƯưẠạẢảẤấẦầẨẩẪẫẬậẮắẰằẲẳẴẵẶặẸẹẺẻẼẽẾếỀềỂểỄễỆệỈỉỊịỌọỎỏỐốỒồỔổỖỗỘộỚớỜờỞởỠỡỢợỤụỦủỨứỪừỬửỮữỰựỲỳỴỵỶỷỸỹ’“”;* $%",
#             'max_batch_size': 8,
#         },
#         'postprocess': {}
#     }
#     common_cfg = {}

#     # Khởi tạo Processor
#     processor = Processor(common_cfg, model_cfg)

#     # Thư mục chứa ảnh
#     folder_path = 'Anh'  # Thay bằng đường dẫn thư mục của bạn

#     # Duyệt và xử lý tất cả ảnh trong thư mục
#     process_images_in_folder(folder_path, processor)

# if __name__ == '__main__':
#     main()
import cv2
import omegaconf
import pdb
from text_detector import TextDetector
from ocr import OCR
from postprocessor import PostProcessor

class Processor:
    def __init__(self, common_cfg, model_cfg):
        self.common_cfg = common_cfg
        self.model_cfg = model_cfg
        self.modules = [
            TextDetector(common_cfg, model_cfg['text_detection']),
            OCR(common_cfg, model_cfg['ocr']),
            PostProcessor(common_cfg, model_cfg['postprocess'])
        ]

    def predict(self, image):
        result = {}
        result['orig_img'] = image
        for module in self.modules:
            result = module.predict(result)
        return result


def main():
    model_cfg = {
        'text_detection': {
            'model_path': 'models/text_detection/epoch=43_val_total_loss=0.66.onnx'
        },
        'ocr': {
            'model_path': 'models/ocr/onnx_print_v4_batch8.onnx',
            'max_sequence_length': 31,
            'charset': "&1#@'()+,-./0123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZ_`abcdefghijklmnopqrstuvwxyz<>ÀÁÂÃÇÈÉÊÌÍÒÓÔÕÙÚÝàáâãçèéêìíòóôõùúýĂăĐđĨĩŨũƠơƯưẠạẢảẤấẦầẨẩẪẫẬậẮắẰằẲẳẴẵẶặẸẹẺẻẼẽẾếỀềỂểỄễỆệỈỉỊịỌọỎỏỐốỒồỔổỖỗỘộỚớỜờỞởỠỡỢợỤụỦủỨứỪừỬửỮữỰựỲỳỴỵỶỷỸỹ’“”;* $%",
            'max_batch_size': 8,
        },
        'postprocess': {}
    }
    common_cfg = {}
    processor = Processor(common_cfg, model_cfg)
    img_fp = 'R1.png'
    img = cv2.imread(img_fp)
    result = processor.predict(img)
    pdb.set_trace()
    print(result)


if __name__ == '__main__':
    main()