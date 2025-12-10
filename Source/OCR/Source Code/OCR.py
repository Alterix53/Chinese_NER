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
            
            # 1. Parse Identifiers
            # Metadata: ID (HVNH), VOLUME (112 or "")
            meta_id = metadata.get("ID", "UNK")
            volume = metadata.get("VOLUME", "").strip()
            
            # Parse 'q01' from 'a_q01'
            # Assumption: pdf_name format is {name}_q{number}
            # Regex to find 'q' followed by digits or hyphens (e.g. q01, q01-02)
            match = re.search(r'_q([0-9\-]+)', pdf_name)
            if match:
                book_num = match.group(1) # '01' or '01-02'
            else:
                # Fallback if specific naming convention not met
                book_num = "00"

            # Construct File ID: HVNH_01
            file_id_val = f"{meta_id}_{book_num}"
            
            # Construct Volume Suffix: 112 or 000
            vol_digits = "".join(filter(str.isdigit, volume))
            if not vol_digits: 
                vol_digits = "000"
            else:
                # Pad to 3 chars if it looks like a number? 
                # LSE_01_04 example in prompt has "04". 
                # HVNH_01.112 has "112".
                # I will just use what I found, maybe pad to 3 if < 3?
                if len(vol_digits) < 3:
                    vol_digits = vol_digits.zfill(3)

            # SECT ID: HVNH_01.112
            sect_id_val = f"{file_id_val}.{vol_digits}"

            # 2. Read Raw JSON
            with open(raw_json_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
            
            # 3. Sort Text Blocks (Vertical: Right-to-Left -> Descending X)
            # Secondary sort: Top-to-Bottom -> Ascending Y
            def sort_key(item):
                c = get_centroid(item["points"])
                return (-c[0], c[1]) # -x, y
            
            sorted_data = sorted(raw_data, key=sort_key)
            
            # 4. Fill Data Structure
            output_obj = copy.deepcopy(format_template)
            
            # Fill FILE info
            f_node = output_obj["FILE"]
            f_node["ID"] = file_id_val
            
            # Fill meta (copy from metadata.json)
            # format.json has keys: TITLE, VOLUME, AUTHOR, PERIOD, LANGUAGE, SOURCE
            # metadata.json has same keys + ID. 
            # User said: "meta: lấy từ metadata.json (bỏ qua ID)"
            for k in f_node["meta"]:
                if k in metadata:
                    f_node["meta"][k] = metadata[k]
                    
            # Fill SECT
            s_node = f_node["SECT"]
            s_node["ID"] = sect_id_val
            s_node["NAME"] = pdf_name 
            
            # Fill PAGES
            # raw_data corresponds to ONE page.
            # format.json "PAGES" is a list. We are processing file-by-page.
            # But the requirement usually implies aggregating?
            # User said: "FILE: ... SECT ... PAGES: một list có 1 dict"
            # "giả sử raw json file hiện tại là a_q01_1.json" -> It implies 1 raw file -> 1 output file?
            # User said: "tên của fornmatted json sẽ kết hợp từ ID định danh + mã số quyển + VOLUME"
            # If we process page by page, we would overwrite the file or need to append?
            # User request: "nhiệm vụ lấy các raw json file ... tiến hành dựa trên format.json"
            # "PAGES: một list có 1 dict ... gồm ID: là số trang sách hiện tại"
            # Creates ambiguity: One huge JSON for the whole book, or 1 JSON per page?
            # User prompt: "giả sử raw json file hiện tại à a_q01_1.json, thì ID là 001"
            # "tên của fornmatted json ... ví dụ LSE_01_04"
            # If I make one file LSE_01_04.json, it should contain ALL pages?
            # But specific logic "PAGES: một list có 1 dict" suggests 1 page per file output?
            # OR the structure only holds 1 page currently.
            # However, filename "LSE_01_04" (Volume level) implies Book Level.
            # If I output per page, I should probably name it `LSE_01_04_001.json`?
            # User comment: "tên của fornmatted json sẽ ..."
            # DOES NOT include page number in user's specified filename "LSE_01_04".
            # This implies the file should contain ALL pages or the user is describing a single-page output that reuses that name (which would overwrite).
            # Re-read: "PAGES: một list có 1 dict".
            # This constraint makes it a valid JSON for exactly 1 page.
            # So I MUST produce 1 JSON file PER page.
            # Therefore I MUST distinguish filenames.
            # I will add the page number to the filename: `[ID]_[Volume]_[Page].json`
            # Wait, user said: "ví dụ a_q01 thì ID là HVNH_01" -> "HVNH_01_112".
            # If I strictly follow "tên của fornmatted json ... là HVNH_01_112", I can't put page number.
            # Maybe the user implies the "SECT" part is the file scope? No, "SECT... PAGES...".
            # Let's look at the example `LSE_01_04.json` provided in context.
            # `file:///c:/.../LSE_01_04.json`
            # Content: "PAGES": [ { "ID": "001", "STC": [...] } ]
            # It only has Page 001.
            # So `LSE_01_04.json` IS the file for Page 1? Or is it a snippet?
            # If Page 2 exists, where does it go?
            # Maybe `LSE_01_04` IS the combined file, but the template only shows 1 page?
            # NO, "PAGES: một list có 1 dict" -> explicitly singular "1 dict".
            # Conclusion: The user likely wants per-page files.
            # I will assume the user-provided name is the PREFIX or he forgot page number.
            # BUT, to be safe and avoid overwriting, I will use: `[ID]_[Volume]_[PageID].json`.
            # OR, perhaps `[OriginalName]_formatted.json`?
            # No, user gave specific naming rule.
            # I'll stick to user rule + Page suffix to ensure uniqueness, or better:
            # Create a folder `[ID]_[Volume]` and put `[PageID].json` inside?
            # User comment: "Save the formatted JSON to a new directory (e.g., .../data/output/a_q01/final/)"
            # User comment 2: "tên của formatted json sẽ ..."
            # Let's save as `.../final/[ID]_[Volume]_[PageID].json`.
            # Actually, looking at `LSE_01_04.json` content again:
            # ` "ID": "LSE_001.004.001.01" ` -> `ID.Volume.Page.Line`.
            # Page ID in file is "001".
            # I will generate specific file names `[ID]_[Volume]_[Page].json` to be safe.
            # If I am wrong, it's safer than overwriting.
            
            page_str = str(page_index + 1).zfill(3)
            
            # STC list
            stc_list = []
            col_idx = 1
            for item in sorted_data:
                col_str = str(col_idx).zfill(2)
                # Text ID: SECT_ID + "." + PAGE_ID + "." + COL_ID
                # SECT_ID = HVNH_01.112
                # PAGE_ID = 001
                # Result: HVNH_01.112.001.01
                txt_id = f"{sect_id_val}.{page_str}.{col_str}"
                
                stc_entry = {
                    "ID": txt_id,
                    "text": item.get("transcription", "")
                }
                stc_list.append(stc_entry)
                col_idx += 1
                
            # Assign to PAGES
            # Requirement: "PAGES: một list có 1 dict"
            page_obj = {
                "ID": page_str,
                "STC": stc_list
            }
            s_node["PAGES"] = [page_obj]
            
            # Save
            # Output folder: .../data/output/a_q01/final/
            # Ensure separate folder for formatted results
            final_dir = os.path.dirname(raw_json_path).replace('json', 'final') # sibling to json
            if not os.path.exists(final_dir):
                # Try to put it in data/output/a_q01/final
                # raw_json_path is .../data/output/a_q01/json/a_q01_1.json
                # split up to parent
                base_out = os.path.dirname(os.path.dirname(raw_json_path)) # .../data/output/a_q01
                final_dir = os.path.join(base_out, "final")
            
            os.makedirs(final_dir, exist_ok=True)
            
            # Filename: HVNH_01_112_001.json (Adding page to avoid overwrite)
            final_name = f"{file_id_val}_{vol_digits}_{page_str}.json"
            final_path = os.path.join(final_dir, final_name)
            
            with open(final_path, 'w', encoding='utf-8') as f:
                json.dump(output_obj, f, ensure_ascii=False, indent=4)
                            
        except Exception as e:
            print(f"[Thread-4] Error: {e}")
        finally:
            queue_in.task_done()

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
