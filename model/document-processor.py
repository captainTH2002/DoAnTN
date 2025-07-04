import os
import json
import cv2
import torch
import yaml
import numpy as np
from PIL import Image
from pathlib import Path
from bpemb import BPEmb
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch_geometric.nn import RGCNConv, FiLMConv, GATv2Conv
from shutil import copyfile

# Import các class từ file OCR đầu tiên
from new_ocr import TextDetector, OCR, PostProcessor, to_json  # Giả sử file đầu tiên tên là first_file.py

# Danh sách nhãn
all_label_list = {
    'congvan': ['text', 'coquan-c1', 'coquan-c2', 'no', 'diadanh', 'thoigian', 'loai', 'noidung-1', 'noidung-2', 'noinhan', 'chucvu', 'ten'],
}

# BaseGraphModel và các mô hình GNN (RGCN, FiLM, GATv2) từ file thứ hai
class BaseGraphModel(pl.LightningModule):
    def __init__(self, general_cfg, model_cfg, n_classes):
        super().__init__()
        self.general_cfg = general_cfg
        self.model_cfg = model_cfg
        self.init_common_layers(general_cfg, model_cfg, n_classes)

    def init_common_layers(self, general_cfg, model_cfg, n_classes):
        self.x_embedding = nn.Embedding(num_embeddings=general_cfg['model']['emb_range']+1, embedding_dim=general_cfg['model']['emb_dim'])
        self.y_embedding = nn.Embedding(num_embeddings=general_cfg['model']['emb_range']+1, embedding_dim=general_cfg['model']['emb_dim'])
        self.w_embedding = nn.Embedding(num_embeddings=general_cfg['model']['emb_range']+1, embedding_dim=general_cfg['model']['emb_dim'])
        self.h_embedding = nn.Embedding(num_embeddings=general_cfg['model']['emb_range']+1, embedding_dim=general_cfg['model']['emb_dim'])
        self.linear_prj = nn.Linear(in_features=general_cfg['model']['text_feature_dim'], out_features=general_cfg['model']['emb_dim']*6)

    def calc_gnn_input(self, x_indexes, y_indexes, text_features):
        left_emb = self.x_embedding(x_indexes[:, 0])
        right_emb = self.x_embedding(x_indexes[:, 1])
        w_emb = self.w_embedding(x_indexes[:, 2])
        top_emb = self.y_embedding(y_indexes[:, 0])
        bot_emb = self.y_embedding(y_indexes[:, 1])
        h_emb = self.h_embedding(y_indexes[:, 2])
        pos_emb = torch.concat([left_emb, right_emb, w_emb, top_emb, bot_emb, h_emb], dim=-1)
        return pos_emb + self.linear_prj(text_features)

class RGCN_Model(BaseGraphModel):
    def __init__(self, general_cfg, model_cfg, n_classes):
        super().__init__(general_cfg, model_cfg, n_classes)
        self.gnn_layers = nn.ModuleList([
            RGCNConv(general_cfg['model']['emb_dim']*6, model_cfg['channels'][0], num_relations=4)
        ])
        for i in range(len(model_cfg['channels'])-1):
            self.gnn_layers.append(RGCNConv(model_cfg['channels'][i], model_cfg['channels'][i+1], num_relations=4))
        self.classifier = nn.Linear(model_cfg['channels'][-1], n_classes)
        self.dropout_layers = nn.ModuleList([nn.Dropout(i) for i in np.linspace(0, general_cfg['model']['dropout_rate'], num=len(self.gnn_layers))])

    def forward(self, x_indexes, y_indexes, text_features, edge_index, edge_type):
        x = self.calc_gnn_input(x_indexes, y_indexes, text_features)
        for layer, dropout_layer in zip(self.gnn_layers, self.dropout_layers):
            x = layer(x, edge_index.to(torch.int64), edge_type)
            x = F.relu(x)
            x = dropout_layer(x)
        logits = self.classifier(x)
        return logits

# Hàm tiện ích từ file thứ hai
def get_bb_from_poly(poly, img_w, img_h):
    x1, y1, x2, y2, x3, y3, x4, y4 = poly
    xmin = max(0, min(x1, x2, x3, x4, img_w))
    xmax = max(0, min(max(x1, x2, x3, x4), img_w))
    ymin = max(0, min(y1, y2, y3, y4, img_h))
    ymax = max(0, min(max(y1, y2, y3, y4), img_h))
    return xmin, ymin, xmax, ymax

def sort_json(json_data):
    bbs, texts = [], []
    for shape in json_data['shapes']:
        x1, y1 = shape['points'][0]
        x2, y2 = shape['points'][1]
        x3, y3 = shape['points'][2]
        x4, y4 = shape['points'][3]
        bb = tuple(int(i) for i in (x1, y1, x2, y2, x3, y3, x4, y4))
        bbs.append(bb)
        texts.append(shape['text'])
    bb2text = dict(zip(bbs, texts))
    bb2idx_original = {x: idx for idx, x in enumerate(bbs)}
    rbbs = row_bbs(bbs.copy())
    sorted_bbs = [bb for row in rbbs for bb in row]
    bb2idx_sorted = {tuple(x): idx for idx, x in enumerate(sorted_bbs)}
    sorted_indices = [bb2idx_sorted[bb] for bb in bb2idx_original.keys()]
    return bb2text, rbbs, bb2idx_sorted, sorted_indices

def row_bbs(bbs):
    bbs.sort(key=lambda x: min(x[0], x[2], x[4], x[6]))
    clusters, y_min = [], []
    for tgt_node in bbs:
        if not clusters:
            clusters.append([tgt_node])
            y_min.append(tgt_node[1])
            continue
        matched = None
        tgt_7_1 = tgt_node[7] - tgt_node[1]
        min_tgt_0_6 = min(tgt_node[0], tgt_node[6])
        max_tgt_2_4 = max(tgt_node[2], tgt_node[4])
        max_left_tgt = min(tgt_node[0], tgt_node[2], tgt_node[4], tgt_node[6])
        for idx, clt in enumerate(clusters):
            src_node = clt[-1]
            src_5_3 = src_node[5] - src_node[3]
            max_src_2_4 = max(src_node[2], src_node[4])
            min_src_0_6 = min(src_node[0], src_node[6])
            overlap_y = (src_5_3 + tgt_7_1) - (max(src_node[5], tgt_node[7]) - min(src_node[3], tgt_node[1]))
            overlap_x = (max_src_2_4 - min_src_0_6) + (max_tgt_2_4 - min_tgt_0_6) - (max(max_src_2_4, max_tgt_2_4) - min(min_src_0_6, min_tgt_0_6))
            if overlap_y > 0.5 * min(src_5_3, tgt_7_1) and overlap_x < 0.6 * min(max_src_2_4 - min_src_0_6, max_tgt_2_4 - min_tgt_0_6):
                distance = max_left_tgt - max(src_node[0], src_node[2], src_node[4], src_node[6])
                if matched is None or distance < matched[1]:
                    matched = (idx, distance)
        if matched is None:
            clusters.append([tgt_node])
            y_min.append(tgt_node[1])
        else:
            clusters[matched[0]].append(tgt_node)
    zip_clusters = list(zip(clusters, y_min))
    zip_clusters.sort(key=lambda x: x[1])
    return [x[0] for x in zip_clusters]

def get_manual_text_feature(text):
    import re
    feature = []
    feature.append(int(re.search('(\d{1,2})\/(\d{1,2})\/(\d{4})', text) is not None))
    feature.append(int(re.search('(\d{1,2}):(\d{1,2})', text) is not None))
    feature.append(int(re.search('^\d+$', text) is not None and len(text) > 5))
    feature.append(int(re.search('^\d{1,3}(\,\d{3})*(\,00)+$', text.replace('.', ',')) is not None or re.search('^\d{1,3}(\,\d{3})+$', text.replace('.', ',')) is not None))
    feature.append(int(text.startswith('-') and re.search('^[\d(\,)]+$', text[1:].replace('.', ',')) is not None and len(text) >= 3))
    feature.append(int(text.isupper()))
    feature.append(int(text.istitle()))
    feature.append(int(text.islower()))
    feature.append(int(re.search('^[A-Z0-9]+$', text) is not None))
    feature.append(int(re.search('^\d+$', text) is not None))
    feature.append(int(re.search('^[a-zA-Z]+$', text) is not None))
    feature.append(int(re.search('^[a-zA-Z0-9]+$', text) is not None))
    feature.append(int(re.search('^[\d|\-|\'|,|\(|\)|.|\/|&|:|+|~|*|\||_|>|@|%]+$', text) is not None))
    feature.append(int(re.search('^[a-zA-Z|\-|\'|,|\(|\)|.|\/|&|:|+|~|*|\||_|>|@|%]+$', text) is not None))
    return feature

def get_input_from_json(json_data, img_w, img_h, word_encoder, use_emb, emb_range):
    x_indexes, y_indexes, text_features = [], [], []
    bb2text, rbbs, bbs2idx_sorted, sorted_indices = sort_json(json_data)
    edges = []
    for row_idx, rbb in enumerate(rbbs):
        for bb_idx_in_row, bb in enumerate(rbb):
            text = bb2text[bb]
            bb_text_feature = get_manual_text_feature(text) + list(np.sum(word_encoder.embed(text), axis=0))
            text_features.append(bb_text_feature)
            xmin, ymin, xmax, ymax = get_bb_from_poly(bb, img_w, img_h)
            if use_emb:
                x_index = [int(xmin * emb_range / img_w), int(xmax * emb_range / img_w), int((xmax - xmin) * emb_range / img_w)]
                y_index = [int(ymin * emb_range / img_h), int(ymax * emb_range / img_h), int((ymax - ymin) * emb_range / img_h)]
            else:
                x_index = [float(xmin / img_w), float(xmax / img_w), float((xmax - xmin) / img_w)]
                y_index = [float(ymin / img_h), float(ymax / img_h), float((ymax - ymin) / img_h)]
            x_indexes.append(x_index)
            y_indexes.append(y_index)
            right_node = rbb[bb_idx_in_row + 1] if bb_idx_in_row < len(rbb) - 1 else None
            if right_node:
                edges.append([bbs2idx_sorted[bb], bbs2idx_sorted[right_node], 1])
                edges.append([bbs2idx_sorted[right_node], bbs2idx_sorted[bb], 2])
            left_node = rbb[bb_idx_in_row - 1] if bb_idx_in_row > 0 else None
            if left_node:
                edges.append([bbs2idx_sorted[bb], bbs2idx_sorted[left_node], 2])
                edges.append([bbs2idx_sorted[left_node], bbs2idx_sorted[bb], 1])
            max_x_overlap, above_node = -1e9, None
            if row_idx > 0:
                for prev_bb in rbbs[row_idx - 1]:
                    xmax_prev_bb = max(prev_bb[2], prev_bb[4])
                    xmin_prev_bb = min(prev_bb[0], prev_bb[6])
                    x_overlap = (xmax_prev_bb - xmin_prev_bb) + (xmax - xmin) - (max(xmax_prev_bb, xmax) - min(xmin_prev_bb, xmin))
                    if x_overlap > max_x_overlap:
                        max_x_overlap = x_overlap
                        above_node = prev_bb
            if above_node:
                edges.append([bbs2idx_sorted[bb], bbs2idx_sorted[above_node], 4])
                edges.append([bbs2idx_sorted[above_node], bbs2idx_sorted[bb], 3])
            max_x_overlap, below_node = -1e9, None
            if row_idx < len(rbbs) - 1:
                for next_bb in rbbs[row_idx + 1]:
                    xmax_next_bb = max(next_bb[2], next_bb[4])
                    xmin_next_bb = min(next_bb[0], next_bb[2])
                    x_overlap = (xmax_next_bb - xmin_next_bb) + (xmax - xmin) - (max(xmax_next_bb, xmax) - min(xmin_next_bb, xmin))
                    if x_overlap > max_x_overlap:
                        max_x_overlap = x_overlap
                        below_node = next_bb
            if below_node:
                edges.append([bbs2idx_sorted[bb], bbs2idx_sorted[below_node], 3])
                edges.append([bbs2idx_sorted[below_node], bbs2idx_sorted[bb], 4])
    edges = torch.tensor(edges, dtype=torch.int32)
    edges = torch.unique(edges, dim=0)
    edge_index, edge_type = edges[:, :2], edges[:, -1]
    return (
        torch.tensor(x_indexes, dtype=torch.int if use_emb else torch.float),
        torch.tensor(y_indexes, dtype=torch.int if use_emb else torch.float),
        torch.tensor(text_features, dtype=torch.float),
        edge_index.t().to(torch.int64),
        edge_type,
    )

def load_model(general_cfg, model_cfg, n_classes, ckpt_path):
    SUPPORTED_MODEL = {'rgcn': RGCN_Model}
    model_type = general_cfg['options']['model_type']
    if model_type not in SUPPORTED_MODEL:
        raise ValueError(f'Model type {model_type} is not supported yet')
    model = SUPPORTED_MODEL[model_type].load_from_checkpoint(
        checkpoint_path=ckpt_path, general_cfg=general_cfg, model_cfg=model_cfg, n_classes=n_classes
    )
    return model

# Pipeline chính
class DocumentProcessor:
    def __init__(self, text_detector_path, ocr_model_path, charset, ckpt_path):
        self.text_detector = TextDetector(text_detector_path)
        self.ocr = OCR(ocr_model_path, charset, max_sequence_length=31, max_batch_size=8)
        self.post_processor = PostProcessor()
        self.ckpt_path = ckpt_path
        self.ckpt_dir = Path(ckpt_path).parent
        with open(os.path.join(self.ckpt_dir, 'train_cfg.yaml'), 'r') as f:
            self.general_cfg = yaml.load(f, Loader=yaml.FullLoader)
        with open(os.path.join(self.ckpt_dir, 'model_cfg.yaml'), 'r') as f:
            self.model_cfg = yaml.load(f, Loader=yaml.FullLoader)
        self.word_encoder = BPEmb(**self.general_cfg['options']['word_encoder'])
        self.label_list = all_label_list['congvan']
        self.model = load_model(self.general_cfg, self.model_cfg, len(self.label_list), ckpt_path)
        self.model.eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def process_image(self, image_path, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
        # Đọc ảnh
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error loading {image_path}")
            return None
        img_h, img_w = img.shape[:2]

        # Text Detection và OCR
        text_detection_result = self.text_detector.predict(img)
        raw_words = self.ocr.predict(text_detection_result['boxes_image'])
        text_result = self.post_processor.process(text_detection_result['coords'], raw_words)

        # Tạo JSON tạm thời
        json_data = {
            'shapes': [],
            'imagePath': image_path,
            'imageHeight': img_h,
            'imageWidth': img_w
        }
        for char, coord in zip(raw_words, text_detection_result['coords']):
            points = [[float(coord[i]), float(coord[i + 1])] for i in range(0, len(coord), 2)]
            json_data['shapes'].append({
                'text': char,
                'label': 'text',
                'points': points,
                'shape_type': 'polygon',
                'flags': {}
            })

        # Tạo input cho GNN
        x_indexes, y_indexes, text_features, edge_index, edge_type = get_input_from_json(
            json_data, img_w, img_h, self.word_encoder, self.general_cfg['options']['use_emb'], self.general_cfg['model']['emb_range']
        )
        x_indexes, y_indexes, text_features, edge_index, edge_type = (
            x_indexes.to(self.device), y_indexes.to(self.device), text_features.to(self.device),
            edge_index.to(self.device), edge_type.to(self.device)
        )

        # Suy luận với GNN
        with torch.no_grad():
            out = self.model(x_indexes, y_indexes, text_features, edge_index, edge_type)
            preds = torch.argmax(out, dim=-1).cpu().numpy()

        # Sắp xếp và cập nhật nhãn
        _, _, _, sorted_indices = sort_json(json_data)
        pred_labels = [self.label_list[preds[i]] for i in sorted_indices]
        for i, shape in enumerate(json_data['shapes']):
            json_data['shapes'][i]['label'] = pred_labels[i]

        # Lưu kết quả
        output_json_path = os.path.join(output_dir, os.path.splitext(os.path.basename(image_path))[0] + '.json')
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=4)
        copyfile(image_path, os.path.join(output_dir, os.path.basename(image_path)))

        print(f"Processed {image_path}. Results saved to {output_dir}")
        return {'text': text_result, 'json_path': output_json_path}

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Document Processing Pipeline')
    parser.add_argument('--image_path', type=str, required=True, help='Path to input image')
    parser.add_argument('--text_detector', type=str, default="./models/text_detection/epoch=43_val_total_loss=0.66.onnx", help='Path to text detection model')
    parser.add_argument('--ocr', type=str, default="./models/ocr/onnx_print_v4_batch8.onnx", help='Path to OCR model')
    parser.add_argument('--ckpt_path', type=str, default="./models/exp6_rgcn_7_layers_augment/model-epoch=98-train_loss=0.173-val_loss=0.150-val_f1=0.997.ckpt", help='Path to GNN checkpoint')
    parser.add_argument('--output_dir', type=str, default="output", help='Output directory')
    args = parser.parse_args()

    charset = "&1#@'()+,-./0123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZ_`abcdefghijklmnopqrstuvwxyz<>ÀÁÂÃÇÈÉÊÌÍÒÓÔÕÙÚÝàáâãçèéêìíòóôõùúýĂăĐđĨĩŨũƠơƯưẠạẢảẤấẦầẨẩẪẫẬậẮắẰằẲẳẴẵẶặẸẹẺẻẼẽẾếỀềỂểỄễỆệỈỉỊịỌọỎỏỐốỒồỔổỖỗỘộỚớỜờỞởỠỡỢợỤụỦủỨứỪừỬửỮữỰựỲỳỴỵỶỷỸỹ'"";* $%"
    processor = DocumentProcessor(args.text_detector, args.ocr, charset, args.ckpt_path)
    result = processor.process_image(args.image_path, args.output_dir)
    if result:
        print(f"Extracted text:\n{result['text']}")

if __name__ == "__main__":
    main()