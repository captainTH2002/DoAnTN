import cv2
from copy import deepcopy
from pathlib import Path
import os
import numpy as np
from shapely.geometry import Polygon
import pdb


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
    zip_clusters = list(zip(clusters, y_min))
    zip_clusters.sort(key=lambda x: x[1])
    zip_clusters = list(np.array(zip_clusters, dtype=object)[:, 0])
    return zip_clusters


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