import cv2
import numpy as np
import pytesseract
import easyocr
from flask import Flask, request, jsonify, render_template
import base64
import os
from werkzeug.utils import secure_filename
import io
from PIL import Image
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Explicitly set Tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Initialize Flask app
app = Flask(__name__)

# Configure upload and cache directories
UPLOAD_FOLDER = 'uploads'
CACHE_FOLDER = 'cache'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CACHE_FOLDER, exist_ok=True)

class OptimizedDocumentScanner:
    def __init__(self):
        """
        Initialize scanner with advanced detection and optimization capabilities
        """
        self.reader = easyocr.Reader(['en'])
        self.cache = {}  # In-memory cache
        self.max_cache_size = 50  # Limit cache size
    
    def compress_image(self, image_path, max_size=(1024, 1024)):
        """
        Compress and resize image for efficient processing
        """
        try:
            with Image.open(image_path) as img:
                # Resize maintaining aspect ratio
                img.thumbnail(max_size, Image.LANCZOS)
                
                # Create compressed file path
                compressed_path = os.path.join(CACHE_FOLDER, f"compressed_{os.path.basename(image_path)}")
                img.save(compressed_path, optimize=True, quality=85)
                
                return compressed_path
        except Exception as e:
            logger.error(f"Image compression error: {e}")
            return image_path

    def preprocess_image(self, image):
        """
        Advanced preprocessing with performance optimizations
        """
        start_time = time.time()
        
        # Convert to grayscale with reduced color depth
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Fast adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            gray, 
            255, 
            cv2.ADAPTIVE_THRESH_MEAN_C, 
            cv2.THRESH_BINARY_INV, 
            11, 
            2
        )
        
        # Quick noise reduction
        denoised = cv2.fastNlMeansDenoising(thresh, None, 10, 7, 21)
        
        processing_time = time.time() - start_time
        logger.info(f"Image preprocessing time: {processing_time:.4f} seconds")
        
        return denoised

    def extract_text(self, image, timeout=10):
        """
        Optimized text extraction with multiple strategies and timeout
        """
        start_time = time.time()
        
        try:
            # Try EasyOCR first with timeout
            results = self.reader.readtext(image, timeout=timeout)
            text = ' '.join([result[1] for result in results])
            
            # Fallback to Tesseract if EasyOCR fails
            if not text.strip():
                text = pytesseract.image_to_string(image)
            
            extraction_time = time.time() - start_time
            logger.info(f"Text extraction time: {extraction_time:.4f} seconds")
            
            return text
        except Exception as e:
            logger.error(f"Text extraction error: {e}")
            return "Could not extract text"

    def process_document(self, image_path):
        """
        Comprehensive document processing with caching and optimization
        """
        # Check cache first
        if image_path in self.cache:
            logger.info("Returning cached result")
            return self.cache[image_path]
        
        # Compress image
        compressed_path = self.compress_image(image_path)
        
        # Read image
        original = cv2.imread(compressed_path)
        
        # Preprocess
        preprocessed = self.preprocess_image(original)
        
        # Extract text
        text = self.extract_text(preprocessed)
        
        # Cache result
        result = (original, text)
        self._update_cache(image_path, result)
        
        return result

    def _update_cache(self, key, value):
        """
        Manage cache with size limit
        """
        if len(self.cache) >= self.max_cache_size:
            # Remove oldest item
            self.cache.pop(next(iter(self.cache)))
        
        self.cache[key] = value

# Create scanner instance
scanner = OptimizedDocumentScanner()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/scan', methods=['POST'])
def scan_document():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        image = request.files['image']
        
        if image.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        # Save temporarily
        filename = secure_filename(image.filename)
        image_path = os.path.join(UPLOAD_FOLDER, filename)
        image.save(image_path)
        
        # Process document
        processed_image, extracted_text = scanner.process_document(image_path)
        
        # Convert image to base64
        _, buffer = cv2.imencode('.jpg', processed_image)
        processed_image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Optional: remove temporary files
        os.remove(image_path)
        
        return jsonify({
            'image': processed_image_base64,
            'text': extracted_text
        })
    
    except Exception as e:
        logger.error(f"Scanning error: {e}")
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
