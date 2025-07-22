import cv2
import omegaconf
import pdb
from .text_detector import TextDetector
from .ocr import OCR

class Processor:
    def __init__(self, common_cfg, model_cfg):
        self.common_cfg = common_cfg
        self.model_cfg = model_cfg
        self.modules = [
            TextDetector(common_cfg, model_cfg['text_detection']),
            OCR(common_cfg, model_cfg['ocr']),
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
            'model_path': ''
        },
        'ocr': {
            'model_path': '',
            'max_sequence_length': 31,
            'charset' : "&1#@'()+,-./0123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZ_`abcdefghijklmnopqrstuvwxyz<>ÀÁÂÃÇÈÉÊÌÍÒÓÔÕÙÚÝàáâãçèéêìíòóôõùúýĂăĐđĨĩŨũƠơƯưẠạẢảẤấẦầẨẩẪẫẬậẮắẰằẲẳẴẵẶặẸẹẺẻẼẽẾếỀềỂểỄễỆệỈỉỊịỌọỎỏỐốỒồỔổỖỗỘộỚớỜờỞởỠỡỢợỤụỦủỨứỪừỬửỮữỰựỲỳỴỵỶỷỸỹ’“”;* $%",
            'max_batch_size': 8,
        }
    }
    common_cfg = {}
    processor = Processor(common_cfg, model_cfg)
    img_fp = 'imgs/layout.jpg'
    img = cv2.imread(img_fp)
    result = processor.predict(img)
    texts = ' '.join(result['ocr']['raw_words'])
    print(texts)


if __name__ == '__main__':
    main()
