import cv2
import numpy as np
from flask import Flask, render_template_string, Response

# Initialize Flask app
app = Flask(__name__)

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Improved spoof detection function using multiple texture analysis methods
def detect_print_attack(gray_frame):
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.2, minNeighbors=7, minSize=(60, 60))

    if len(faces) == 0:
        return "No Face Detected", None

    for (x, y, w, h) in faces:
        face_region = gray_frame[y:y+h, x:x+w]

        # Apply Laplacian variance for texture sharpness analysis
        laplacian_var = cv2.Laplacian(face_region, cv2.CV_64F).var()

        # Apply Sobel edge detection for better texture differentiation
        sobel_edges = np.mean(cv2.Sobel(face_region, cv2.CV_64F, 1, 1, ksize=5))

        # Decision rule: Combining Laplacian and Sobel for improved classification
        if laplacian_var < 15 and sobel_edges < 50:  # More refined threshold values
            return "Spoofing Attack Detected (Print Attack)", (x, y, w, h)

    return "Real Human Face Detected", (x, y, w, h)

# Function to capture and process video feed
def generate_frames():
    cap = cv2.VideoCapture(0)  # Open webcam

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        result, face_coords = detect_print_attack(gray)

        if face_coords:
            x, y, w, h = face_coords
            color = (0, 255, 0) if "Real" in result else (0, 0, 255)  # Green for real, red for spoof
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, result, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

# HTML template for the interface
html_template = '''
<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Facial Anti-Spoofing Detection</title>
  <style>
    body { font-family: Arial, sans-serif; background-color: #007BFF; display: flex; justify-content: center; align-items: center; height: 100vh; margin: 0; }
    .container { background-color: white; padding: 30px; border-radius: 8px; box-shadow: 0 0 15px rgba(0, 0, 0, 0.1); text-align: center; width: 400px; }
    h2 { color: #333; margin-bottom: 20px; }
    img { width: 100%; border-radius: 5px; margin-top: 20px; }
  </style>
</head>
<body>

  <div class="container">
    <h2>Facial Anti-Spoofing Detection</h2>
    <img src="{{ url_for('video_feed') }}" alt="Video Feed">
  </div>

</body>
</html>
'''

@app.route('/')
def index():
    """Render the webcam interface"""
    return render_template_string(html_template)

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
