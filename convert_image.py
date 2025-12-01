import fitz  # PyMuPDF
from PIL import Image
import io
import os

def convert_pdf_page_to_image(pdf_path, page_number, output_image_path, zoom=2.0):
    """
    Converts a specific page of a PDF file to an image file (e.g., PNG, JPG).

    Args:
        pdf_path (str): Path to the input PDF file.
        page_number (int): The page number to convert (0-indexed).
        output_image_path (str): Path where the output image will be saved.
        zoom (float): Zoom factor for higher resolution (2.0 = 200% resolution).
    """
    try:
        # Open the PDF document
        doc = fitz.open(pdf_path)

        # Check if the requested page number is valid
        if 0 <= page_number < len(doc):
            page = doc.load_page(page_number)
        else:
            print(f"Error: Page number {page_number} is out of range.")
            return

        # Define the transformation matrix for higher resolution
        # zoom=2.0 increases the resolution by a factor of 2 (4x the pixels)
        matrix = fitz.Matrix(zoom, zoom)

        # Render the page to a pixmap (raw image data)
        pix = page.get_pixmap(matrix=matrix, alpha=False)

        # Convert the pixmap to a PIL Image object
        # Pixmaps provide the data as bytes in memory (pix.samples)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        # Save the PIL Image to a file
        img.save(output_image_path)

        print(f"Successfully converted page {page_number + 1} to {output_image_path}")

        # Close the document
        doc.close()

    except FileNotFoundError:
        print(f"Error: PDF file not found at {pdf_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

def convert_all_pdf_pages_to_images(pdf_path, output_folder="output_images", format="png", zoom=2.0):
    """
    Converts every page of a PDF file to separate image files.
    """
    try:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        doc = fitz.open(pdf_path)
        matrix = fitz.Matrix(zoom, zoom)

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            # Create the output filename (e.g., page_001.png)
            output_filename = os.path.join(
                output_folder,
                f"page_{page_num + 1:03d}.{format}"
            )
            img.save(output_filename)
            print(f"Saved {output_filename}")

        doc.close()
        print(f"\nAll pages converted and saved in the '{output_folder}' folder.")

    except FileNotFoundError:
        print(f"Error: PDF file not found at {pdf_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

# --- Configuration ---
# convert_all_pdf_pages_to_images('my_document.pdf', 'document_images', format='jpg')


# --- Configuration ---
# NOTE: Replace 'input.pdf' with the actual path to your PDF file
PDF_FILE = 'data/giam_cuong.pdf'
# NOTE: Replace 'output_page_1.png' with your desired output file name and extension
OUTPUT_FILE = 'content/giam_cuong.png'
# Page 0 is the first page
PAGE_TO_CONVERT = 6
# ---------------------

# Execute the conversion
# convert_pdf_page_to_image(PDF_FILE, PAGE_TO_CONVERT, OUTPUT_FILE)
# Since I cannot access your file system, I'll provide the ready function call:
def main():
    convert_pdf_page_to_image(PDF_FILE, PAGE_TO_CONVERT, OUTPUT_FILE)

if __name__ == "__main__":
    main()
# Example of how you would call it:
# convert_pdf_page_to_image('my_document.pdf', 0, 'first_page.jpg', zoom=3.0)