import os
import cv2
import fitz  # PyMuPDF  
import numpy as np
import json
import threading
import queue
import copy
import re

from PIL import Image
from paddleocr import PaddleOCR

QUEUE_MAX_SIZE = 30  
OCR_ENGINE = PaddleOCR(
    text_detection_model_name="PP-OCRv5_mobile_det",
    text_recognition_model_name="PP-OCRv5_mobile_rec",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    device="cpu",
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
            
import pandas as pd

def thread_post_processing(queue_in, metadata_path, format_path):
    # Load Metadata & Format
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        with open(format_path, 'r', encoding='utf-8') as f:
            format_template = json.load(f)
    except Exception as e:
        print(f"[Thread-4] Error loading config files: {e}")
        return

    # Helper to calculate centroid for sorting
    def get_centroid(points):
        # points is typically [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        # or flat list. The code in raw saving uses .tolist()
        pts = np.array(points)
        return np.mean(pts, axis=0)

    # Accumulate all pages: books_data[pdf_name] = [ {page_index:..., data:...}, ... ]
    books_data = {}

    while True:
        task = queue_in.get()
        if task is None:
            queue_in.task_done()
            break
            
        try:
            # Task contains: raw_json_path, pdf_name (e.g. a_q01), page_index
            raw_json_path = task.get("raw_json_path")
            pdf_name = task.get("pdf_name") # e.g. a_q01
            page_index = task.get("page_index") # 0-based
            
            # Read Raw JSON
            if raw_json_path and os.path.exists(raw_json_path):
                with open(raw_json_path, 'r', encoding='utf-8') as f:
                    raw_data = json.load(f)
                
                # Sort Text Blocks (Vertical: Right-to-Left -> Descending X)
                # Secondary sort: Top-to-Bottom -> Ascending Y
                def sort_key(item):
                    c = get_centroid(item["points"])
                    return (-c[0], c[1]) # -x, y
                
                sorted_data = sorted(raw_data, key=sort_key)
                
                if pdf_name not in books_data:
                    books_data[pdf_name] = []
                
                books_data[pdf_name].append({
                    "page_index": page_index,
                    "data": sorted_data,
                    "raw_path": raw_json_path
                })
            
        except Exception as e:
            print(f"[Thread-4] Error processing page task: {e}")
        finally:
            queue_in.task_done()

    # End of queue logic: Process aggregated books and write final JSONs
    print("[Thread-4] Start aggregating and writing output files...")
    
    for pdf_name, pages_list in books_data.items():
        try:
            # Sort pages by index
            pages_list.sort(key=lambda x: x["page_index"])
            
            # 1. Parse Name and Book Number from pdf_name
            # Format expectation: {name}_q{number} -> a_q01
            match = re.search(r'^(.*)_q([0-9\-]+)$', pdf_name)
            if match:
                book_name_val = match.group(1)
                book_num = match.group(2)
            else:
                book_name_val = pdf_name
                book_num = "00"

            # 2. Parse Identifiers
            meta_id = metadata.get("ID", "UNK")
            volume = metadata.get("VOLUME", "").strip()

            if not volume:
                volume = book_num
            
            # Construct File ID: e.g. HVNH_01
            file_id_val = f"{meta_id}_{book_num}"
            
            # Construct SECT ID: e.g. HVNH_01.01 (Volume 01)
            sect_id_val = f"{file_id_val}.{volume}"
            
            # 3. Build Output Structure
            output_obj = copy.deepcopy(format_template)
            
            # FILE level
            f_node = output_obj["FILE"]
            f_node["ID"] = file_id_val
            
            # Meta
            for k in f_node["meta"]:
                if k in metadata:
                    f_node["meta"][k] = metadata[k]
            
            # Explicitly set VOLUME in case it was inferred
            f_node["meta"]["VOLUME"] = volume
            
            # SECT level
            s_node = f_node["SECT"]
            s_node["ID"] = sect_id_val
            s_node["NAME"] = book_name_val
            
            # PAGES level
            output_pages = []
            
            # Prepare Excel Data
            excel_rows = []

            for p_entry in pages_list:
                page_idx = p_entry["page_index"]
                sorted_items = p_entry["data"]
                
                # Page ID 3 digits: 001
                page_str = str(page_idx + 1).zfill(3)
                
                stc_list = []
                col_idx = 1
                for item in sorted_items:
                    col_str = str(col_idx).zfill(2)
                    # Text ID: SECT_ID + "." + PAGE_ID + "." + COL_ID
                    txt_id = f"{sect_id_val}.{page_str}.{col_str}"
                    # content and box
                    text_content = item.get("transcription", "")
                    points = item.get("points", [])
                    
                    stc_entry = {
                        "ID": txt_id,
                        "text": text_content
                    }
                    stc_list.append(stc_entry)
                    
                    # Add to Excel rows
                    excel_rows.append({
                        "ID": txt_id,
                        "Image Name": f"{pdf_name}.png",
                        "Han Char": text_content,
                        "Image Box": str(points)
                    })
                    
                    col_idx += 1
                
                output_pages.append({
                    "ID": page_str,
                    "STC": stc_list
                })
            
            s_node["PAGES"] = output_pages
            
            # 4. Save to Final Files
            if not pages_list:
                continue

            first_raw_path = pages_list[0]["raw_path"]
            base_out = os.path.dirname(os.path.dirname(first_raw_path))
            final_dir = os.path.join(base_out, "final")
            os.makedirs(final_dir, exist_ok=True)
            
            # Save JSON
            final_json_name = f"{pdf_name}.json"
            final_json_path = os.path.join(final_dir, final_json_name)
            
            with open(final_json_path, 'w', encoding='utf-8') as f:
                json.dump(output_obj, f, ensure_ascii=False, indent=4)
                
            # Save Excel
            if excel_rows:
                df = pd.DataFrame(excel_rows)
                final_xlsx_name = f"{pdf_name}.xlsx"
                final_xlsx_path = os.path.join(final_dir, final_xlsx_name)
                df.to_excel(final_xlsx_path, index=False)
                print(f"[Thread-4] Saved combined JSON: {final_json_path} and Excel: {final_xlsx_path}")
            else:
                print(f"[Thread-4] Saved combined JSON: {final_json_path} (No data for Excel)")

        except Exception as e:
            print(f"[Thread-4] Error aggregating book {pdf_name}: {e}")


def thread_run_ocr(queue_in, output_base_dir, queue_out=None):
    while True:
        task = queue_in.get()
        
        if task is None:
            if queue_out:
                queue_out.put(None)
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
        
            raw_json_path = ""
            if filtered_data:
                # Save JSON
                raw_json_path = os.path.join(out_dir, "json", f"{file_base_name}.json")
                with open(raw_json_path, 'w', encoding='utf-8') as f:
                    json.dump(filtered_data, f, ensure_ascii=False, indent=4)

                # Save Image
                img = img_array.copy()
                for item in filtered_data:
                    box = np.array(item["points"]).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(img, [box], True, color=(0, 255, 0), thickness=2)
                cv2.imwrite(os.path.join(out_dir, "img", f"res_{file_base_name}.jpg"), img)
            
            # Push to Post-Processing Queue
            if queue_out and raw_json_path:
                task_post = {
                    "raw_json_path": raw_json_path,
                    "pdf_name": pdf_name,
                    "page_index": page_idx
                }
                queue_out.put(task_post)

        except Exception as e:
            print(f"[Thread-3] Error processing page {task.get('page_index')}: {e}")
        finally:
            queue_in.task_done()
            
if __name__ == "__main__":
    # Setup directories
    base_dir = os.getcwd()
    raw_dir = os.path.join(base_dir, 'data', 'raw')
    process_dir = os.path.join(base_dir, 'data', 'processed')
    output_base_dir = os.path.join(base_dir, 'data', 'output')
    
    pdf_files = [os.path.join(raw_dir, f) for f in os.listdir(raw_dir) if f.endswith(".pdf")]
    if not pdf_files:
        print("No PDF files found in directory")
        exit()

    # Threading 
    q1_convert_to_preprocess = queue.Queue(maxsize=QUEUE_MAX_SIZE)
    q2_preprocess_to_ocr = queue.Queue(maxsize=QUEUE_MAX_SIZE)
    q3_ocr_to_post = queue.Queue(maxsize=QUEUE_MAX_SIZE)

    # Load Meta/Format paths
    metadata_path = os.path.join(base_dir, 'metadata.json')
    format_path = os.path.join(base_dir, 'format.json')

    t1 = threading.Thread(target=thread_convert_pdf, args=(pdf_files, q1_convert_to_preprocess, process_dir))
    t2 = threading.Thread(target=thread_preprocess, args=(q1_convert_to_preprocess, q2_preprocess_to_ocr, False))
    t3 = threading.Thread(target=thread_run_ocr, args=(q2_preprocess_to_ocr, output_base_dir, q3_ocr_to_post))
    t4 = threading.Thread(target=thread_post_processing, args=(q3_ocr_to_post, metadata_path, format_path))

    t1.start()
    t2.start()
    t3.start()
    t4.start()

    t1.join()
    t2.join()
    t3.join()
    t4.join()

    print("Done")
