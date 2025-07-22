import os
import sys
import cv2
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
    text_detection_model_path = r"C:\Do_An\backend\text_detect_ocr\models\text_detection\epoch=43_val_total_loss=0.66.onnx"
    ocr_model_path = r"C:\Do_An\backend\text_detect_ocr\models\ocr\onnx_print_v4_batch8.onnx"

    model_cfg = {
        'text_detection': {'model_path': text_detection_model_path},
        'ocr': {
            'model_path': ocr_model_path,
            'max_sequence_length': 31,
            'charset': "&1#@'()+,-./0123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZ_`abcdefghijklmnopqrstuvwxyz<>ÀÁÂÃÇÈÉÊÌÍÒÓÔÕÙÚÝàáâãçèéêìíòóôõùúýĂăĐđĨĩŨũƠơƯưẠạẢảẤấẦầẨẩẪẫẬậẮắẰằẲẳẴẵẶặẸẹẺẻẼẽẾếỀềỂểỄễỆệỈỉỊịỌọỎỏỐốỒồỔổỖỗỘộỚớỜờỞởỠỡỢợỤụỦủỨứỪừỬửỮữỰựỲỳỴỵỶỷỸỹ’“”;* $%",
            'max_batch_size': 8,
        },
        'postprocess': {}
    }
    common_cfg = {}

    if len(sys.argv) < 2:
        print("Usage: python img2text.py <image_path>", file=sys.stderr)
        sys.exit(1)

    img_fp = sys.argv[1]
    img = cv2.imread(img_fp)
    if img is None:
        print(f"ERROR: Cannot read image at {img_fp}", file=sys.stderr)
        sys.exit(2)

    processor = Processor(common_cfg, model_cfg)
    result = processor.predict(img)

    if isinstance(result, dict):
        extracted_text = result.get('text', '').strip()
    else:
        extracted_text = str(result).strip()

    # Viết ra file cùng thư mục ảnh
    output_path = os.path.join(os.path.dirname(img_fp), "ocr_result.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(extracted_text)

if __name__ == '__main__':
    main()
