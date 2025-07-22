import onnxruntime as ort
from copy import deepcopy
import numpy as np
import cv2
from PIL import Image
import pdb


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
        text_output = np.concatenate(text_output, axis=0)
        text_output = text_output[:batch_images_length]  # shape (num_boxes, max_sequence_length, num_classes)
        raw_words = [self.index_to_word(prob) for prob in text_output]

        assert len(raw_words) == len(result['text_detection']['boxes_image'])
        result['ocr'] = {}
        result['ocr']['raw_words'] = raw_words
        return result