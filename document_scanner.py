import cv2
import numpy as np
import pytesseract
import easyocr
from flask import Flask, request, jsonify, render_template
import base64
import os
from werkzeug.utils import secure_filename

# Explicitly set Tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Initialize Flask app BEFORE creating the scanner
app = Flask(__name__)

class UniversalDocumentScanner:
    def __init__(self):
        """
        Initialize scanner with advanced detection capabilities
        """
        self.reader = easyocr.Reader(['en'])

    def preprocess_image(self, image):
        """
        Advanced preprocessing for various paper sizes and qualities
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply multiple preprocessing techniques
        
        # 1. Enhanced adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            gray, 
            255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 
            11, 
            2
        )
        
        # 2. Noise reduction
        denoised = cv2.fastNlMeansDenoising(thresh, None, 10, 7, 21)
        
        return denoised

    def detect_document_edges(self, image):
        """
        Advanced document edge detection for multiple paper sizes
        """
        # Find contours with multiple strategies
        contours, _ = cv2.findContours(
            image, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Sort contours by area, largest first
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        # Filter and approximate contours
        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            
            # Look for quadrilateral shape (document)
            if len(approx) == 4:
                return approx
        
        return None

    def perspective_transform(self, original, edges):
        """
        Perspective transform supporting various paper sizes
        """
        # Flatten and order points
        pts = edges.reshape(4, 2)
        rect = np.zeros((4, 2), dtype="float32")
        
        # Top-left will have smallest sum, bottom-right largest
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        
        # Top-right smallest difference, bottom-left largest
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        
        # Compute width and height dynamically
        (tl, tr, br, bl) = rect
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        
        # Define destination points
        dst = np.array([ 
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")
        
        # Compute perspective transform
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(original, M, (maxWidth, maxHeight))
        
        return warped

    def extract_text(self, image):
        """
        Advanced text extraction supporting multiple formats
        """
        # Multiple OCR strategies
        try:
            # Try EasyOCR first
            results = self.reader.readtext(image)
            text = ' '.join([result[1] for result in results])
            
            # If no text, fallback to Tesseract
            if not text.strip():
                text = pytesseract.image_to_string(image)
            
            return text
        except Exception as e:
            # Final fallback
            return pytesseract.image_to_string(image)

    def process_document(self, image_path):
        """
        Comprehensive document processing
        """
        # Read image
        original = cv2.imread(image_path)
        
        # Preprocess
        preprocessed = self.preprocess_image(original)
        
        # Detect edges
        edges = self.detect_document_edges(preprocessed)
        
        if edges is not None:
            # Transform perspective
            warped = self.perspective_transform(original, edges)
            
            # Extract text
            text = self.extract_text(warped)
            
            return warped, text
        
        return original, "No document detected"

# Create scanner instance AFTER app is initialized
scanner = UniversalDocumentScanner()

# Ensure upload folder exists
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/scan', methods=['POST'])
def scan_document():
    try:
        # Get image from request
        if 'image' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        image = request.files['image']
        
        # Check if filename is empty
        if image.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        # Save temporarily
        filename = secure_filename(image.filename)
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image.save(image_path)
        
        # Process document
        processed_image, extracted_text = scanner.process_document(image_path)
        
        # Convert image to base64 for frontend
        _, buffer = cv2.imencode('.jpg', processed_image)
        processed_image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Optional: remove temporary file
        os.remove(image_path)
        
        # First return the image, then the text
        return jsonify({
            'image': processed_image_base64,
            'text': extracted_text
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # Ensure uploads and other necessary directories exist
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True)
