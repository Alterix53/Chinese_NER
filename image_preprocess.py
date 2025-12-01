import cv2
import numpy as np
from PIL import Image

def preprocess_for_ocr(image_path, output_path="preprocessed_image.png"):
    """
    Thực hiện tiền xử lý cơ bản cho ảnh chữ Hán (Chuyển xám, Khử nhiễu, Ngưỡng).
    """
    try:
        # 1. Đọc ảnh
        img = cv2.imread(image_path)
        if img is None:
            print(f"Lỗi: Không thể đọc tệp ảnh tại đường dẫn: {image_path}")
            return None

        # 2. Chuyển về Thang độ Xám (Grayscale)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #print("Đã chuyển ảnh sang thang độ xám.")

        # 3. Khử Nhiễu (Denoising) - Sử dụng Gaussian Blur nhẹ hơn
        # Kernel (3, 3) giúp làm mượt mà không làm mất nét quá nhiều
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        #print("Đã áp dụng Gaussian Blur nhẹ để khử nhiễu.")

        # 4. Cân bằng Ngưỡng (Thresholding) - Sử dụng Otsu để giảm độ mạnh xử lý
        # Otsu tự động tìm ngưỡng tối ưu dựa trên histogram và bớt khắc nghiệt hơn adaptive threshold
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        #print("Đã áp dụng Thresholding Otsu.")

        # Lưu ảnh đã tiền xử lý
        cv2.imwrite(output_path, binary)
        #print(f"Ảnh đã xử lý được lưu tại: {output_path}")
        
        return output_path

    except Exception as e:
        print(f"Đã xảy ra lỗi trong quá trình xử lý: {e}")
        return None

# # --- CÁCH SỬ DỤNG ---
# input_file = 'anh_chu_han_goc.jpg'  # Thay thế bằng đường dẫn tệp ảnh của bạn
# preprocessed_file = preprocess_for_ocr(input_file) 
# # Bây giờ bạn có thể dùng 'preprocessed_file' để chạy OCR (ví dụ: Tesseract)

DEFAULT_INPUT_FILE = 'content/giam_cuong.png'
DEFAULT_OUTPUT_FILE = 'processed/giam_cuong_preprocessed.png'

def main():
    user_input = input(f"Nhập đường dẫn file ảnh đầu vào (bấm Enter để dùng mặc định: '{DEFAULT_INPUT_FILE}'): ").strip()
    user_output = input(f"Nhập đường dẫn file ảnh đầu ra (bấm Enter để dùng mặc định: '{DEFAULT_OUTPUT_FILE}'): ").strip()

    input_file = user_input if user_input else DEFAULT_INPUT_FILE
    output_file = user_output if user_output else DEFAULT_OUTPUT_FILE

    path = preprocess_for_ocr(input_file, output_file)
    if path:
        print(f"Đã lưu ảnh đã tiền xử lý tại: {path}")
    else:
        print("Quá trình xử lý thất bại.")

if __name__ == "__main__":
    main()