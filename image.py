from PIL import Image
import pytesseract

# Set the Tesseract command location (adjust the path accordingly)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCRtesseract.exe'

def extract_text_from_image(image_path, language='eng'):
    try:
        # Open the image file
        img = Image.open(image_path)

        # Perform OCR using Tesseract
        text = pytesseract.image_to_string(img, lang=language)

        return text.strip()

    except Exception as e:
        print(f"Error: {e}")
        return None

# Example usage:
image_path = r'C:\Users\Abdul Sattar\Downloads\RealTime-OCR-master\RealTime-OCR-master\WhatsApp Image 2023-12-19 at 15.18.48_8726b29b.jpg'
extracted_text = extract_text_from_image(image_path)

if extracted_text:
    print(f"Extracted Text: {extracted_text}")
else:
    print("Text extraction failed.")
