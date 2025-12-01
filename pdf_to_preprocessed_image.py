import os
from typing import Optional

import cv2
import fitz  # PyMuPDF
import numpy as np
from PIL import Image


# Easily configurable default PDF path.
INPUT_PDF_PATH = os.path.join("data", "giam_cuong.pdf")


def convert_pdf_to_image(pdf_path: str, page_num: int, zoom: float = 2.0) -> np.ndarray:
    """
    Convert a specific PDF page to an OpenCV-friendly image array.

    Args:
        pdf_path: Path to the PDF file.
        page_num: Zero-based page index to convert.
        zoom: Magnification factor for higher resolution renders.

    Returns:
        np.ndarray: Image in BGR color space suitable for OpenCV pipelines.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Không tìm thấy tệp PDF tại: {pdf_path}")

    if zoom <= 0:
        raise ValueError("Giá trị zoom phải lớn hơn 0.")

    doc = fitz.open(pdf_path)
    try:
        if not (0 <= page_num < len(doc)):
            raise ValueError(
                f"Số trang không hợp lệ. PDF có {len(doc)} trang, "
                f"nhưng bạn yêu cầu trang {page_num + 1}."
            )

        page = doc.load_page(page_num)
        matrix = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=matrix, alpha=False)

        # Convert PyMuPDF pixmap to PIL image, then to numpy for OpenCV.
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        rgb_array = np.array(img)
        bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
        return bgr_array
    finally:
        doc.close()


def preprocess_image_for_ocr(image_data: np.ndarray) -> np.ndarray:
    """
    Apply grayscale, Gaussian blur, and adaptive thresholding to enhance OCR.

    Args:
        image_data: Input image array in BGR color space.

    Returns:
        np.ndarray: Binarized image ready for OCR.
    """
    if image_data is None:
        raise ValueError("Dữ liệu ảnh đầu vào không hợp lệ.")

    gray = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    binary = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        21,
        8,
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    strengthened = cv2.dilate(binary, kernel, iterations=1)
    return strengthened


def _prompt_page_number(total_pages: int) -> Optional[int]:
    """Ask the user for a 1-based page number and convert to zero-based index."""
    try:
        user_input = input(
            f"Nhập số trang muốn chuyển đổi (1-{total_pages}): "
        ).strip()
        if not user_input:
            print("Bạn chưa nhập số trang.")
            return None

        page_number = int(user_input)
        page_index = page_number - 1
        if not (0 <= page_index < total_pages):
            print("Số trang nằm ngoài phạm vi cho phép.")
            return None
        return page_index
    except ValueError:
        print("Giá trị trang không hợp lệ. Vui lòng nhập số nguyên.")
        return None


def main() -> None:
    """CLI entry point for the PDF -> preprocessed image workflow."""
    pdf_path = INPUT_PDF_PATH
    if not os.path.exists(pdf_path):
        print(f"Lỗi: Không tìm thấy tệp PDF tại: {pdf_path}")
        return

    try:
        with fitz.open(pdf_path) as doc:
            total_pages = len(doc)
    except Exception as exc:
        print(f"Lỗi khi mở PDF: {exc}")
        return

    page_index = _prompt_page_number(total_pages)
    if page_index is None:
        return

    doc_name = os.path.splitext(os.path.basename(pdf_path))[0]
    page_label = page_index + 1

    content_dir = "content"
    processed_dir = "Processed"
    os.makedirs(content_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    raw_filename = f"{doc_name}_{page_label}.png"
    raw_output_path = os.path.join(content_dir, raw_filename)
    processed_output_path = os.path.join(processed_dir, raw_filename)

    try:
        image_data = convert_pdf_to_image(pdf_path, page_index, zoom=2.0)
        processed_image = preprocess_image_for_ocr(image_data)

        # Save raw export before preprocessing.
        if not cv2.imwrite(raw_output_path, image_data):
            raise IOError("Không thể lưu ảnh gốc trước khi xử lý.")

        # Save processed image to the Processed directory.
        if not cv2.imwrite(processed_output_path, processed_image):
            raise IOError("Không thể lưu ảnh đầu ra.")

        print(
            "Đã xử lý xong:\n"
            f"- Ảnh gốc: {os.path.abspath(raw_output_path)}\n"
            f"- Ảnh tiền xử lý: {os.path.abspath(processed_output_path)}"
        )
    except FileNotFoundError as fnf_err:
        print(f"Lỗi: {fnf_err}")
    except ValueError as val_err:
        print(f"Lỗi: {val_err}")
    except IOError as io_err:
        print(f"Lỗi khi lưu ảnh: {io_err}")
    except Exception as exc:
        print(f"Đã xảy ra lỗi không xác định: {exc}")


if __name__ == "__main__":
    main()

