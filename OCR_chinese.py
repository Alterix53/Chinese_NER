

import os
from typing import Iterable, List, Tuple

from paddleocr import PaddleOCR

# Default to the latest processed output; feel free to change this path.
DEFAULT_IMAGE_PATH = os.path.join("Processed", "giam_cuong_preprocessed.png")

# Initialize PaddleOCR once to avoid re-loading weights on every call.
OCR_ENGINE = PaddleOCR(
    text_detection_model_name="PP-OCRv5_mobile_det",
    text_recognition_model_name="PP-OCRv5_mobile_rec",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    device="gpu", # đổi lại cpu nếu chưa tải phiên bản của gpu
)


def run_ocr(image_path: str) -> List[Tuple[str, float]]:
    """
    Execute PaddleOCR on the given image.

    Args:
        image_path: Path to the image file to be processed.

    Returns:
        A list of (text, confidence) tuples extracted from the image.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Không tìm thấy ảnh đầu vào: {image_path}")

    result = OCR_ENGINE.predict(image_path)

    for res in result:  
        res.print()  
        res.save_to_img("output/OCR_result")  # lưu ra ảnh
        res.save_to_json("output/OCR_json") # lưu ra json file
        
    return result


def run_folder(folder_path: str) -> None:
    """
    Run OCR over every supported image inside the provided folder.
    """
    if not os.path.isdir(folder_path):
        raise NotADirectoryError(f"Không tìm thấy thư mục đầu vào: {folder_path}")

    supported_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    image_files = []
    for name in sorted(os.listdir(folder_path)):
        path = os.path.join(folder_path, name)
        if not os.path.isfile(path):
            continue
        if os.path.splitext(name)[1].lower() not in supported_exts:
            continue
        image_files.append(path)

    if not image_files:
        print("Không tìm thấy ảnh hợp lệ trong thư mục.")
        return

    for image_path in image_files:
        print(f"==> Đang xử lý: {image_path}")
        try:
            run_ocr(image_path)
        except Exception as exc:
            print(f"Lỗi khi chạy OCR cho '{image_path}': {exc}")

def print_results(entries: Iterable[Tuple[str, float]]) -> None:
    """Pretty-print OCR results."""
    for idx, (text, confidence) in enumerate(entries, start=1):
        print(f"{idx:02d}. ({confidence:.2%}) {text}")


def main() -> None:
    mode = input("Chọn chế độ OCR: 1) Single ảnh  2) Folder ảnh [1/2]: ").strip() or "1"

    try:
        if mode == "2":
            folder_path = input("Nhập đường dẫn folder chứa ảnh: ").strip()
            if not folder_path:
                print("Đường dẫn folder không được bỏ trống.")
                return
            run_folder(folder_path)
        else:
            prompt = (
                f"Nhập đường dẫn ảnh cần OCR (Enter để dùng '{DEFAULT_IMAGE_PATH}'): "
            )
            image_path = input(prompt).strip() or DEFAULT_IMAGE_PATH
            results = run_ocr(image_path)
            if not results:
                print("Không tìm thấy văn bản nào trong ảnh.")
                return

    except Exception as exc:
        print(f"Đã xảy ra lỗi khi chạy PaddleOCR: {exc}")


if __name__ == "__main__":
    main()
