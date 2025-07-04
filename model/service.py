# pipeline.py
import sys
from pathlib import Path
from new_ocr import main as ocr_main  # Import main từ file Text Detection + OCR
from graph import main as inference_main  # Import main từ file Graph Inference

def run_pipeline(image_input):
    # Bước 1: Chạy Text Detection và OCR
    print("Step 1: Running Text Detection and OCR...")
    ocr_main(image_input)

    # Bước 2: Chạy Graph-based Inference
    print("Step 2: Running Graph-based Inference...")
    inference_main(image_input)

    print(f"Pipeline completed for {image_input}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pipeline.py <image_path_or_directory>")
        sys.exit(1)
    
    input_path = sys.argv[1]
    run_pipeline(input_path)