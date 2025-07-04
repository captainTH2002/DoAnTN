from fastapi import FastAPI, File, UploadFile
import uvicorn
import cv2
import numpy as np
import os
from new_ocr import Processor  # Import class Processor từ new_ocr.py

app = FastAPI()

# Khởi tạo model OCR một lần khi start server
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
processor = Processor(common_cfg, model_cfg)

@app.post("/ocr")
async def ocr_api(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return {"error": "Cannot read image"}
    result = processor.predict(img)
    # Ghép text lại thành 1 chuỗi
    text = "\n".join(result['ocr']['raw_words'])
    return {"text": text}

if __name__ == "__main__":
    uvicorn.run("ocr_api:app", host="0.0.0.0", port=8002, reload=True)
