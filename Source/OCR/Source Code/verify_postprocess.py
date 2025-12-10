import json
import os
import re
import copy
import glob
import sys

# Constants
BASE_DIR = os.getcwd()
LOG_FILE = "verify_log.txt"

def log(msg):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
    print(msg)

def verify_logic():
    log("Starting verification...")
    RAW_JSON_DIR = os.path.join(BASE_DIR, 'data', 'output', 'a_q01', 'json')
    METADATA_PATH = os.path.join(BASE_DIR, 'metadata.json')
    FORMAT_PATH = os.path.join(BASE_DIR, 'format.json')

    if not os.path.exists(METADATA_PATH):
        log(f"Metadata missing at {METADATA_PATH}")
        return

    json_files = glob.glob(os.path.join(RAW_JSON_DIR, "*.json"))
    log(f"Found {len(json_files)} raw json files.")

    with open(METADATA_PATH, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    with open(FORMAT_PATH, 'r', encoding='utf-8') as f:
        format_template = json.load(f)

    for raw_json_path in json_files:
        try:
            filename = os.path.basename(raw_json_path)
            log(f"Processing {filename}")
            
            # Logic
            parts = filename.rsplit('_', 1)
            if len(parts) != 2: continue
            
            pdf_name = parts[0]
            page_str = parts[1].replace('.json', '')
            try:
                page_index = int(page_str) - 1
            except:
                page_index = 0
            
            # Meta parsing
            meta_id = metadata.get("ID", "UNK")
            volume = metadata.get("VOLUME", "").strip()
            
            match = re.search(r'_q([0-9\-]+)', pdf_name)
            book_num = match.group(1) if match else "00"
            file_id_val = f"{meta_id}_{book_num}"
            
            vol_digits = "".join(filter(str.isdigit, volume))
            if not vol_digits: vol_digits = "000"
            elif len(vol_digits) < 3: vol_digits = vol_digits.zfill(3)
            
            sect_id_val = f"{file_id_val}.{vol_digits}"
            
            with open(raw_json_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
            
            # Sort without numpy
            # each item has "points": [[x,y]...]
            def get_centroid(item):
                points = item["points"]
                # Assume list of lists
                n = len(points)
                if n == 0: return (0,0)
                sum_x = sum(p[0] for p in points)
                sum_y = sum(p[1] for p in points)
                return (sum_x/n, sum_y/n)
                
            filtered_data = [] # remove low confidence? handled in OCR.py, this reads OCR output
            
            sorted_data = sorted(raw_data, key=lambda x: (-get_centroid(x)[0], get_centroid(x)[1]))
            
            # output
            output_obj = copy.deepcopy(format_template)
            f_node = output_obj["FILE"]
            f_node["ID"] = file_id_val
            for k in f_node["meta"]:
                if k in metadata: f_node["meta"][k] = metadata[k]
            
            s_node = f_node["SECT"]
            s_node["ID"] = sect_id_val
            s_node["NAME"] = pdf_name
            
            page_fmt = str(page_index + 1).zfill(3)
            stc_list = []
            col_idx = 1
            for item in sorted_data:
                col_str = str(col_idx).zfill(2)
                txt_id = f"{sect_id_val}.{page_fmt}.{col_str}"
                stc_list.append({
                    "ID": txt_id,
                    "text": item.get("transcription", "")
                })
                col_idx += 1
            
            s_node["PAGES"] = [{
                "ID": page_fmt,
                "STC": stc_list
            }]
            
            # Save
            final_dir = os.path.dirname(raw_json_path).replace('json', 'final')
            if not os.path.exists(final_dir):
                base_out = os.path.dirname(os.path.dirname(raw_json_path))
                final_dir = os.path.join(base_out, 'final')
            os.makedirs(final_dir, exist_ok=True)
            
            final_name = f"{file_id_val}_{vol_digits}_{page_fmt}.json"
            final_path = os.path.join(final_dir, final_name)
            
            with open(final_path, 'w', encoding='utf-8') as f:
                json.dump(output_obj, f, ensure_ascii=False, indent=4)
            log(f"Saved {final_path}")
            
        except Exception as e:
            log(f"Error on {raw_json_path}: {e}")

if __name__ == "__main__":
    try:
        verify_logic()
    except Exception as e:
        log(f"Fatal: {e}")
