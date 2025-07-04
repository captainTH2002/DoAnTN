import os
import re
import json
import torch
import yaml
import math
import numpy as np
import unidecode
from PIL import Image
from pathlib import Path
from bpemb import BPEmb
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
from torch_geometric.nn import RGCNConv, FiLMConv, GATv2Conv


all_label_list = {
        # 'hoadongtgt': ['text', 'total_payment', 'tax_rate', 'sub_total', 'amount', 'quantity', 'unit', 'description', 'unit_price', 'tax_code', 'address', 'payment_method', 'company', 'buyer_name', 'date', 'no', 'account_no', 'serial', 'tel'],
        'congvan':['text', 'coquan-c1', 'coquan-c2', 'no', 'diadanh', 'thoigian', 'loai', 'noidung-1', 'noidung-2', 'noinhan', 'chucvu', 'ten'],
                        }

# Character mapping for Vietnamese text
uc = {
    'a':'a', 'á':'a', 'à':'a', 'ả':'a', 'ã':'a', 'ạ':'a',
    'ă':'a', 'ắ':'a', 'ằ':'a', 'ẳ':'a', 'ẵ':'a', 'ặ':'a',
    'â':'a', 'ấ':'a', 'ầ':'a', 'ẩ':'a', 'ẫ':'a', 'ậ':'a',
    'e':'e', 'é':'e', 'è':'e', 'ẻ':'e', 'ẽ':'e', 'ẹ':'e',
    'ê':'e', 'ế':'e', 'ề':'e', 'ể':'e', 'ễ':'e', 'ệ':'e',
    'i':'i', 'í':'i', 'ì':'i', 'ỉ':'i', 'ĩ':'i', 'ị':'i',
    'o':'o', 'ó':'o', 'ò':'o', 'ỏ':'o', 'õ':'o', 'ọ':'o',
    'ô':'o', 'ố':'o', 'ồ':'o', 'ổ':'o', 'ỗ':'o', 'ộ':'o',
    'ơ':'o', 'ớ':'o', 'ờ':'o', 'ở':'o', 'ỡ':'o', 'ợ':'o',
    'u':'u', 'ú':'u', 'ù':'u', 'ủ':'u', 'ũ':'u', 'ụ':'u',
    'ư':'u', 'ứ':'u', 'ừ':'u', 'ử':'u', 'ữ':'u', 'ự':'u',
    'y':'y', 'ý':'y', 'ỳ':'y', 'ỷ':'y', 'ỹ':'y', 'ỵ':'y',
    'đ':'d'
}

# Base Model Class
class BaseGraphModel(pl.LightningModule):
    def __init__(self, general_cfg, model_cfg, n_classes):
        super().__init__()
        self.general_cfg = general_cfg
        self.mode_config = model_cfg
        
        self.init_common_layers(general_cfg, model_cfg, n_classes)
        self.init_lightning_stuff(general_cfg, n_classes)

    def init_common_layers(self, general_cfg, model_cfg, n_classes):
        self.x_embedding = nn.Embedding(num_embeddings=general_cfg['model']['emb_range']+1, embedding_dim=general_cfg['model']['emb_dim'])
        self.y_embedding = nn.Embedding(num_embeddings=general_cfg['model']['emb_range']+1, embedding_dim=general_cfg['model']['emb_dim'])
        self.w_embedding = nn.Embedding(num_embeddings=general_cfg['model']['emb_range']+1, embedding_dim=general_cfg['model']['emb_dim'])
        self.h_embedding = nn.Embedding(num_embeddings=general_cfg['model']['emb_range']+1, embedding_dim=general_cfg['model']['emb_dim'])
        self.linear_prj = nn.Linear(in_features=general_cfg['model']['text_feature_dim'], out_features=general_cfg['model']['emb_dim']*6)

    def init_lightning_stuff(self, general_cfg, n_classes):
        self.criterion = nn.CrossEntropyLoss(label_smoothing=general_cfg['training']['label_smoothing'])
        self.train_f1 = torchmetrics.F1Score(task='multiclass', threshold=0.5, num_classes=n_classes)
        self.val_f1 = torchmetrics.F1Score(task='multiclass', threshold=0.5, num_classes=n_classes)

    def configure_optimizers(self):
        base_lr = self.general_cfg['training']['base_lr']
        opt = torch.optim.AdamW(self.parameters(), lr=base_lr, weight_decay=self.general_cfg['training']['weight_decay'])

        if self.general_cfg['training']['use_warmup']:
            num_warmpup_epoch = self.general_cfg['training']['warmup_ratio'] * self.general_cfg['training']['num_epoch']
            def lr_foo(epoch):
                if epoch <= num_warmpup_epoch:
                    return 0.75 ** (num_warmpup_epoch - epoch)
                else:
                    return 0.97 ** (epoch - num_warmpup_epoch)
            
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer=opt,
                lr_lambda=lr_foo
            )
            return [opt], [scheduler]
        
        return opt
    
    def calc_gnn_input(self, x_indexes, y_indexes, text_features):
        # Calculate spatial position embedding
        left_emb = self.x_embedding(x_indexes[:, 0])    # (n_nodes, embed_size)
        right_emb = self.x_embedding(x_indexes[:, 1])
        w_emb = self.w_embedding(x_indexes[:, 2])
        top_emb = self.y_embedding(y_indexes[:, 0])
        bot_emb = self.y_embedding(y_indexes[:, 1])
        h_emb = self.h_embedding(y_indexes[:, 2])
        pos_emb = torch.concat([left_emb, right_emb, w_emb, top_emb, bot_emb, h_emb], dim=-1)

        return pos_emb + self.linear_prj(text_features)
    
    def common_step(self, batch, batch_idx):
        x_indexes, y_indexes, text_features, edge_index, edge_type, labels = batch
        logits = self.forward(x_indexes, y_indexes, text_features, edge_index, edge_type)
        loss = self.criterion(logits, labels)
        return logits, loss, labels
    
    def training_step(self, batch, batch_idx):
        logits, loss, labels = self.common_step(batch, batch_idx)
        self.train_f1(torch.argmax(logits, dim=-1), labels)
        self.log('train_loss', loss, on_step=True, on_epoch=False, prog_bar=False)
        self.log('train_f1', self.train_f1, on_step=True, on_epoch=False, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        logits, loss, labels = self.common_step(batch, batch_idx)
        self.val_f1(torch.argmax(logits, dim=-1), labels)
        self.log_dict({
            'val_loss': loss,
            'val_f1': self.val_f1
        }, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_start(self) -> None:
        print('\n')

# RGCN Model
class RGCN_Model(BaseGraphModel):
    def __init__(self, general_cfg, model_cfg, n_classes):
        super().__init__(general_cfg, model_cfg, n_classes)
        self.init_gnn_layers(general_cfg, model_cfg, n_classes)
    
    def init_gnn_layers(self, general_cfg, model_cfg, n_classes):
        self.gnn_layers = nn.ModuleList([
            RGCNConv(
                in_channels=general_cfg['model']['emb_dim']*6, 
                out_channels=model_cfg['channels'][0], 
                num_relations=4
            )
        ])
        for i in range(len(model_cfg['channels'])-1):
            self.gnn_layers.append(
                RGCNConv(
                    in_channels=model_cfg['channels'][i], 
                    out_channels=model_cfg['channels'][i+1], 
                    num_relations=4
                )
            )
        self.classifier = nn.Linear(in_features=model_cfg['channels'][-1], out_features=n_classes)
        self.dropout_layers = nn.ModuleList([
            nn.Dropout(i) for i in np.linspace(0, self.general_cfg['model']['dropout_rate'], num=len(self.gnn_layers))
        ])
    
    def forward(self, x_indexes, y_indexes, text_features, edge_index, edge_type):
        x = self.calc_gnn_input(x_indexes, y_indexes, text_features)
        for layer, dropout_layer in zip(self.gnn_layers, self.dropout_layers):
            x = layer(x, edge_index.to(torch.int64), edge_type)
            x = F.relu(x)
            x = dropout_layer(x)
        logits = self.classifier(x)
        return logits

# GNN FiLM Model
class GNN_FiLM_Model(BaseGraphModel):
    def __init__(self, general_cfg, model_cfg, n_classes):
        super().__init__(general_cfg, model_cfg, n_classes)
        self.init_gnn_layers(general_cfg, model_cfg, n_classes)
    
    def init_gnn_layers(self, general_cfg, model_cfg, n_classes):
        self.gnn_layers = nn.ModuleList([
            FiLMConv(
                in_channels=general_cfg['model']['emb_dim']*6, 
                out_channels=model_cfg['channels'][0], 
                num_relations=4,
                act=None
            )
        ])
        for i in range(len(model_cfg['channels'])-1):
            self.gnn_layers.append(
                FiLMConv(
                    in_channels=model_cfg['channels'][i], 
                    out_channels=model_cfg['channels'][i+1], 
                    num_relations=4,
                    act=None
                )
            )
        self.classifier = nn.Linear(in_features=model_cfg['channels'][-1], out_features=n_classes)
    
    def forward(self, x_indexes, y_indexes, text_features, edge_index, edge_type):
        x = self.calc_gnn_input(x_indexes, y_indexes, text_features)
        for layer in self.gnn_layers:
            x = layer(x, edge_index.to(torch.int64), edge_type)
            x = F.relu(x)
        x = F.dropout(x, p=self.general_cfg['model']['dropout_rate'])
        logits = self.classifier(x)
        return logits

# GATv2 Model
class GATv2_Model(BaseGraphModel):
    def __init__(self, general_cfg, model_cfg, n_classes):
        super().__init__(general_cfg, model_cfg, n_classes)
        self.init_gnn_layers(general_cfg, model_cfg, n_classes)
    
    def init_gnn_layers(self, general_cfg, model_cfg, n_classes):
        self.gnn_layers = nn.ModuleList([
            GATv2Conv(
                in_channels=general_cfg['model']['emb_dim']*6, 
                out_channels=model_cfg['channels'][0], 
                heads=model_cfg['num_heads'],
                dropout=model_cfg['attn_dropout'],
                concat=model_cfg['concat']
            )
        ])
        for i in range(len(model_cfg['channels'])-1):
            self.gnn_layers.append(
                GATv2Conv(
                    in_channels=model_cfg['channels'][i] if not model_cfg['concat'] else model_cfg['channels'][i]*model_cfg['num_heads'], 
                    out_channels=model_cfg['channels'][i+1], 
                    heads=model_cfg['num_heads'],
                    dropout=model_cfg['attn_dropout'],
                    concat=model_cfg['concat']
                )
            )
        self.classifier = nn.Linear(
            in_features=model_cfg['channels'][-1] if not model_cfg['concat'] else model_cfg['channels'][-1]*model_cfg['num_heads'], 
            out_features=n_classes
        )
    
    def forward(self, x_indexes, y_indexes, text_features, edge_index, edge_type):
        x = self.calc_gnn_input(x_indexes, y_indexes, text_features)
        for layer in self.gnn_layers:
            x = layer(x, edge_index.to(torch.int64))
            x = F.relu(x)
        x = F.dropout(x, p=self.general_cfg['model']['dropout_rate'])
        logits = self.classifier(x)
        return logits

# Utility functions from my_utils.py
def remove_accent(text):
    return unidecode.unidecode(text)

def get_img_fp_from_json_fp(json_fp):
    """Get the image file path from a JSON file path."""
    ls_ext = ['.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG']
    for ext in ls_ext:
        img_fp = json_fp.with_suffix(ext)
        if img_fp.exists():
            return img_fp
    return None

def get_bb_from_poly(poly, img_w, img_h):
    """Get the bounding box from a polygon."""
    x1, y1, x2, y2, x3, y3, x4, y4 = poly    # tl -> tr -> br -> bl
    xmin = min(x1, x2, x3, x4)
    xmin = max(0, min(xmin, img_w))
    xmax = max(x1, x2, x3, x4)
    xmax = max(0, min(xmax, img_w))
    ymin = min(y1, y2, y3, y4)
    ymin = max(0, min(ymin, img_h))
    ymax = max(y1, y2, y3, y4)
    ymax = max(0, min(ymax, img_h))
    return xmin, ymin, xmax, ymax

def max_left(bb):
    """Get the leftmost x-coordinate of a polygon."""
    return min(bb[0], bb[2], bb[4], bb[6])

def max_right(bb):
    """Get the rightmost x-coordinate of a polygon."""
    return max(bb[0], bb[2], bb[4], bb[6])

def row_bbs(bbs):
    """Group bounding boxes into rows."""
    bbs.sort(key=lambda x: max_left(x))
    clusters, y_min = [], []
    for tgt_node in bbs:
        if len(clusters) == 0:
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
    if np.ndim(zip_clusters) == 1:
        zip_clusters = list(zip_clusters)
    else:
        zip_clusters = list(np.array(zip_clusters, dtype=object)[:, 0])
    return zip_clusters

def sort_json(json_data):
    """Sort the JSON data by position."""
    bbs, labels, texts = [], [], []
    for shape in json_data['shapes']:
        x1, y1 = shape['points'][0]  # tl
        x2, y2 = shape['points'][1]  # tr
        x3, y3 = shape['points'][2]  # br
        x4, y4 = shape['points'][3]  # bl
        bb = tuple(int(i) for i in (x1,y1,x2,y2,x3,y3,x4,y4))
        bbs.append(bb)
        labels.append(shape['label'])
        try:
            texts.append(shape['text'])
        except:
            texts.append('')

    bb2label = dict(zip(bbs, labels))   # by order in data['shapes']
    bb2text = dict(zip(bbs, texts))
    bb2idx_original = {x: idx for idx, x in enumerate(bbs)}   # by order in data['shapes']
    rbbs = row_bbs(bbs.copy())
    sorted_bbs = [bb for row in rbbs for bb in row]  # left to right, top to bottom
    bb2idx_sorted = {tuple(x): idx for idx, x in enumerate(sorted_bbs)}   # left to right, top to bottom
    sorted_indices = [bb2idx_sorted[bb] for bb in bb2idx_original.keys()]

    return bb2label, bb2text, rbbs, bb2idx_sorted, sorted_indices

def get_manual_text_feature(text):
    """Extract manual text features."""
    feature = []

    # Is it a date?
    feature.append(int(re.search('(\d{1,2})\/(\d{1,2})\/(\d{4})', text) != None))

    # Is it a time?
    feature.append(int(re.search('(\d{1,2}):(\d{1,2})', text) != None))
        
    # Is it a product code?
    feature.append(int(re.search('^\d+$', text) != None and len(text) > 5))

    # Is it a positive currency?
    feature.append(int(re.search('^\d{1,3}(\,\d{3})*(\,00)+$', text.replace('.', ',')) != None or re.search('^\d{1,3}(\,\d{3})+$', text.replace('.', ',')) != None))
    
    # Is it a negative currency?
    feature.append(int(text.startswith('-') and re.search('^[\d(\,)]+$', text[1:].replace('.', ',')) != None and len(text) >= 3))

    # Is it uppercase?
    feature.append(int(text.isupper()))

    # Is it title case?
    feature.append(int(text.istitle()))

    # Is it lowercase?
    feature.append(int(text.islower()))
    
    # Does it contain only uppercase letters and numbers?
    feature.append(int(re.search('^[A-Z0-9]+$', text) != None))

    # Does it contain only numbers?
    feature.append(int(re.search('^\d+$', text) != None))

    # Does it contain only letters?
    feature.append(int(re.search('^[a-zA-Z]+$', text) != None))

    # Does it contain only letters and numbers?
    feature.append(int(re.search('^[a-zA-Z0-9]+$', text) != None))

    # Does it contain only numbers and punctuation?
    feature.append(int(re.search('^[\d|\-|\'|,|\(|\)|.|\/|&|:|+|~|*|\||_|>|@|%]+$', text) != None))

    # Does it contain only letters and punctuation?
    feature.append(int(re.search('^[a-zA-Z|\-|\'|,|\(|\)|.|\/|&|:|+|~|*|\||_|>|@|%]+$', text) != None))

    return feature

def load_model(general_cfg, model_cfg, n_classes, ckpt_path=None):
    """Load a model from a checkpoint."""
    SUPPORTED_MODEL = {
        'rgcn': RGCN_Model,
        'gatv2': GATv2_Model,
        'gnn_film': GNN_FiLM_Model
    }
    
    model_type = general_cfg['options']['model_type']
    if model_type not in SUPPORTED_MODEL:
        raise ValueError(f'Model type {model_type} is not supported yet')
    
    if ckpt_path is not None:
        model = SUPPORTED_MODEL[model_type].load_from_checkpoint(
                    checkpoint_path=ckpt_path,
                    general_cfg=general_cfg, 
                    model_cfg=model_cfg, 
                    n_classes=n_classes
                )
    else:
        model = SUPPORTED_MODEL[model_type](
            general_cfg=general_cfg, 
            model_cfg=model_cfg, 
            n_classes=n_classes
        )
    
    return model

# Functions from new_test.py
def get_input_from_json(json_fp, word_encoder, use_emb, emb_range):
    """Create input graph data from a JSON file."""
    with open(json_fp, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    img_fp = get_img_fp_from_json_fp(json_fp)
    if img_fp is None:
        raise FileNotFoundError(f"Cannot find image file for {json_fp}")
    img_w, img_h = Image.open(img_fp).size

    x_indexes, y_indexes, text_features = [], [], []
    bb2label, bb2text, rbbs, bbs2idx_sorted, sorted_indices = sort_json(json_data)

    edges = []
    for row_idx, rbb in enumerate(rbbs):
        for bb_idx_in_row, bb in enumerate(rbb):
            # Process text features
            text = bb2text[bb]
            if word_encoder.lang != 'vi':
                text = ''.join(c for c in text if ord(c) < 128)  # Remove non-ASCII if not Vietnamese
            bb_text_feature = get_manual_text_feature(text) + list(np.sum(word_encoder.embed(text), axis=0))
            text_features.append(bb_text_feature)

            # Process geometric features
            xmin, ymin, xmax, ymax = get_bb_from_poly(bb, img_w, img_h)
            if use_emb:
                x_index = [int(xmin * emb_range / img_w), int(xmax * emb_range / img_w), int((xmax - xmin) * emb_range / img_w)]
                y_index = [int(ymin * emb_range / img_h), int(ymax * emb_range / img_h), int((ymax - ymin) * emb_range / img_h)]
            else:
                x_index = [float(xmin / img_w), float(xmax / img_w), float((xmax - xmin) / img_w)]
                y_index = [float(ymin / img_h), float(ymax / img_h), float((ymax - ymin) / img_h)]
            x_indexes.append(x_index)
            y_indexes.append(y_index)

            # Build edges
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

def inference_single_image(json_path, ckpt_path, output_dir, mart_name="711"):
    """Suy luận trên một ảnh và file JSON đơn lẻ."""
    # Tạo thư mục đầu ra nếu chưa tồn tại
    os.makedirs(output_dir, exist_ok=True)
    
    # Đường dẫn checkpoint
    ckpt_dir = Path(ckpt_path).parent

    # Đọc cấu hình
    with open(os.path.join(ckpt_dir, 'train_cfg.yaml'), 'r') as f:
        general_cfg = yaml.load(f, Loader=yaml.FullLoader)
    with open(os.path.join(ckpt_dir, 'model_cfg.yaml'), 'r') as f:
        model_cfg = yaml.load(f, Loader=yaml.FullLoader)

    # Khởi tạo word encoder
    word_encoder = BPEmb(**general_cfg['options']['word_encoder'])

    # Lấy danh sách nhãn từ mart_name
    label_list = all_label_list[mart_name]
    use_emb = general_cfg['options']['use_emb']
    emb_range = general_cfg['model']['emb_range']

    # Tải mô hình từ checkpoint
    model = load_model(general_cfg, model_cfg, n_classes=len(label_list), ckpt_path=ckpt_path)
    model.eval()

    # Đọc file JSON
    json_fp = Path(json_path)
    with open(json_fp, 'r', encoding='utf-8') as f:
        json_data = json.load(f)

    # Tạo dữ liệu đầu vào
    x_indexes, y_indexes, text_features, edge_index, edge_type = get_input_from_json(
        json_fp, word_encoder, use_emb, emb_range
    )

    # Chuyển dữ liệu sang CPU hoặc GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x_indexes, y_indexes, text_features, edge_index, edge_type = (
        x_indexes.to(device),
        y_indexes.to(device),
        text_features.to(device),
        edge_index.to(device),
        edge_type.to(device)
    )
    model.to(device)

    # Suy luận
    with torch.no_grad():
        out = model(x_indexes, y_indexes, text_features, edge_index, edge_type)
        preds = torch.argmax(out, dim=-1).cpu().numpy()

    # Sắp xếp dự đoán theo thứ tự ban đầu
    _, _, _, _, sorted_indices = sort_json(json_data)
    pred_labels = [label_list[preds[i]] for i in sorted_indices]

    # Cập nhật nhãn trong JSON
    for i, shape in enumerate(json_data['shapes']):
        json_data['shapes'][i]['label'] = pred_labels[i]

    # Lưu file JSON đã cập nhật
    output_json_path = os.path.join(output_dir, json_fp.name)
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=4)

    # Sao chép ảnh sang thư mục đầu ra
    img_fp = get_img_fp_from_json_fp(json_fp)
    if img_fp:
        from shutil import copyfile
        copyfile(img_fp, os.path.join(output_dir, img_fp.name))

    print(f"Đã xử lý xong {json_fp.name}. Kết quả lưu tại {output_dir}")

if __name__ == "__main__":
    import argparse
    
    # Tạo trình phân tích tham số
    parser = argparse.ArgumentParser(description='Trích xuất thông tin từ ảnh tài liệu')
    parser.add_argument('--json_path', type=str, default="../input/82-TB-VPCP_page_1.json", 
                        help='Đường dẫn đến file JSON đầu vào')
    parser.add_argument('--ckpt_path', type=str, 
                        default="./", 
                        help='Đường dẫn đến file checkpoint mô hình')
    parser.add_argument('--output_dir', type=str, default="../out", 
                        help='Thư mục đầu ra cho kết quả')
    parser.add_argument('--mart_name', type=str, default="congvan", 
                        help='Tên mart (phải khớp với all_label_list)')
    
    # Phân tích tham số
    args = parser.parse_args()
    
    # Kiểm tra xem các file đầu vào có tồn tại không
    if not os.path.exists(args.json_path):
        print(f"Lỗi: File JSON đầu vào không tồn tại: {args.json_path}")
        exit(1)
    
    if not os.path.exists(args.ckpt_path):
        print(f"Lỗi: File checkpoint không tồn tại: {args.ckpt_path}")
        exit(1)
    
    # Tạo thư mục đầu ra nếu chưa tồn tại
    os.makedirs(args.output_dir, exist_ok=True)
    
    # In thông tin đầu vào
    print(f"Xử lý với các tham số:")
    print(f"  - File JSON: {args.json_path}")
    print(f"  - Checkpoint: {args.ckpt_path}")
    print(f"  - Thư mục đầu ra: {args.output_dir}")
    print(f"  - Tên mart: {args.mart_name}")
    
    try:
        # Gọi hàm xử lý
        inference_single_image(args.json_path, args.ckpt_path, args.output_dir, args.mart_name)
        
        # Kiểm tra kết quả
        output_json = os.path.join(args.output_dir, os.path.basename(args.json_path))
        if os.path.exists(output_json):
            print(f"Thành công: File JSON đầu ra đã được tạo tại {output_json}")
        else:
            print(f"Cảnh báo: File JSON đầu ra không tồn tại tại {output_json}")
    
    except Exception as e:
        print(f"Lỗi khi xử lý: {e}")
        import traceback
        traceback.print_exc()