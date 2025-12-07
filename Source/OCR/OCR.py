import os
import cv2
import fitz  # PyMuPDF  
import numpy as np
import json
import threading
import queue

from PIL import Image
from paddleocr import PaddleOCR

QUEUE_MAX_SIZE = 30  
OCR_ENGINE = PaddleOCR(
    text_detection_model_name="PP-OCRv5_mobile_det",
    text_recognition_model_name="PP-OCRv5_mobile_rec",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    device="gpu",
)

def convert_page_to_image(page: fitz.Page, zoom: float = 2.0) -> np.ndarray:
    """
    Convert a specific PDF page to image in BGR color space suitable for OpenCV pipelines.
    """
    matrix = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=matrix, alpha=False)

    # Convert PyMuPDF pixmap to PIL image, then to numpy for OpenCV.
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    rgb_array = np.array(img)
    # Convert RGB to BGR for OpenCV
    bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
    
    return bgr_array

def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    
    return cv2.LUT(image, table)

def preprocess_image_for_ocr(image_data: np.ndarray, preprocess: bool) -> np.ndarray:
    """
    Pipeline: Grayscale -> Deskew -> Erosion
    """
    if not preprocess:
        return image_data 
    
    # Grayscale
    gray = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)

    denoised = cv2.bilateralFilter(gray, 9, 75, 75)

    enhanced = adjust_gamma(denoised, gamma=0.4)

    return enhanced

def thread_convert_pdf(pdf_files, queue_out, process_dir):
    for pdf_path in pdf_files:
        try:
            pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
            output_folder = os.path.join(process_dir, pdf_name)
            os.makedirs(output_folder, exist_ok=True)

            with fitz.open(pdf_path) as doc:
                for page_index, page in enumerate(doc):
                    # Convert
                    image_data = convert_page_to_image(page, zoom=2.0)
                    
                    # Đóng gói
                    task = {
                        "image_data": image_data,
                        "pdf_name": pdf_name,
                        "page_index": page_index,
                        "output_folder": output_folder,
                    }
                    
                    queue_out.put(task)

        except Exception as e:
            print(f"[Thread-1: Convert] Error {pdf_path}: {e}")

    queue_out.put(None)

def thread_preprocess(queue_in, queue_out, preprocess):
    while True:
        task = queue_in.get()
        
        if task is None:
            queue_out.put(None) 
            queue_in.task_done()
            break
        
        try:
            processed_img = preprocess_image_for_ocr(task["image_data"], preprocess)

            task["image_data"] = processed_img
            queue_out.put(task)

            file_name = f"{task['pdf_name']}_{task['page_index'] + 1}.png"
            save_path = os.path.join(task['output_folder'], file_name)
            cv2.imwrite(save_path, processed_img)
            
            
            
        except Exception as e:
            print(f"[Thread-2: Preprocess] Error: {e}")
        finally:
            queue_in.task_done()
            
def thread_run_ocr(queue_in, output_base_dir):
    while True:
        task = queue_in.get()
        
        if task is None:
            queue_in.task_done()
            break
            
        try:
            img_array = task["image_data"] 
            pdf_name = task["pdf_name"]
            page_idx = task["page_index"]
            file_base_name = f"{pdf_name}_{page_idx + 1}"

            out_dir = os.path.join(output_base_dir, pdf_name)
            os.makedirs(os.path.join(out_dir, "img"), exist_ok=True)
            os.makedirs(os.path.join(out_dir, "json"), exist_ok=True)

            result = OCR_ENGINE.predict(img_array)
            
            filtered_data = []
            
            if isinstance(result, list):
                for line in result:
                    if not line: continue 
                    
                    if isinstance(line, dict) and 'dt_polys' in line:
                         boxes = line['dt_polys']
                         texts = line['rec_texts']
                         scores = line['rec_scores']
                         for box, txt, score in zip(boxes, texts, scores):
                            if score > 0.5:
                                filtered_data.append({
                                    "points": box.tolist() if isinstance(box, np.ndarray) else box,
                                    "transcription": txt,
                                    "confidence": float(score)
                                })
                    elif isinstance(line, list) and len(line) == 2 and isinstance(line[1], tuple):
                         box = line[0]
                         text, score = line[1]
                         if score > 0.5:
                            filtered_data.append({
                                "points": box,
                                "transcription": text,
                                "confidence": float(score)
                            })
        
            if filtered_data:
                # Save JSON
                json_path = os.path.join(out_dir, "json", f"{file_base_name}.json")
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(filtered_data, f, ensure_ascii=False, indent=4)

                # Save Image
                img = img_array.copy()
                for item in filtered_data:
                    box = np.array(item["points"]).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(img, [box], True, color=(0, 255, 0), thickness=2)
                cv2.imwrite(os.path.join(out_dir, "img", f"res_{file_base_name}.jpg"), img)

        except Exception as e:
            print(f"[Thread-3] Error processing page {task.get('page_index')}: {e}")
        finally:
            queue_in.task_done()
            
if __name__ == "__main__":
    # Setup directories
    base_dir = os.getcwd()
    raw_dir = os.path.join(base_dir, 'Source', 'OCR', 'data', 'raw')
    process_dir = os.path.join(base_dir, 'Source', 'OCR', 'data', 'processed')
    output_base_dir = os.path.join(base_dir, 'Source', 'OCR', 'data', 'output')
    
    pdf_files = [os.path.join(raw_dir, f) for f in os.listdir(raw_dir) if f.endswith(".pdf")]
    if not pdf_files:
        print("No PDF files found in directory")
        exit()

    # Threading 
    q1_convert_to_preprocess = queue.Queue(maxsize=QUEUE_MAX_SIZE)
    q2_preprocess_to_ocr = queue.Queue(maxsize=QUEUE_MAX_SIZE)

    t1 = threading.Thread(target=thread_convert_pdf, args=(pdf_files, q1_convert_to_preprocess, process_dir))
    t2 = threading.Thread(target=thread_preprocess, args=(q1_convert_to_preprocess, q2_preprocess_to_ocr, False))
    t3 = threading.Thread(target=thread_run_ocr, args=(q2_preprocess_to_ocr, output_base_dir))

    t1.start()
    t2.start()
    t3.start()

    t1.join()
    t2.join()
    t3.join()

    print("Done")  
