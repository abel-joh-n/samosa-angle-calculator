from flask import Flask, render_template, request, redirect, url_for
import cv2
import numpy as np
import os
import math

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def calculate_angle(a, b, c):
    ab = np.array(a) - np.array(b)
    cb = np.array(c) - np.array(b)
    cosine_angle = np.dot(ab, cb) / (np.linalg.norm(ab) * np.linalg.norm(cb))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def process_image(filepath):
    image = cv2.imread(filepath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    triangle = None
    for cnt in contours:
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if len(approx) == 3:
            triangle = approx
            break

    if triangle is None:
        return None, None

    pts = [tuple(point[0]) for point in triangle]
    a, b, c = pts
    angle_A = calculate_angle(b, a, c)
    angle_B = calculate_angle(a, b, c)
    angle_C = calculate_angle(a, c, b)

    # Draw triangle
    cv2.drawContours(image, [triangle], 0, (0, 255, 0), 2)
    cv2.putText(image, f"{angle_A:.1f}°", a, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(image, f"{angle_B:.1f}°", b, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    cv2.putText(image, f"{angle_C:.1f}°", c, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 128, 255), 2)

    output_path = os.path.join(UPLOAD_FOLDER, 'output.jpg')
    cv2.imwrite(output_path, image)

    return output_path, [angle_A, angle_B, angle_C]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)  # ✅ ensures the folder exists

            file.save(path)
            output_path, angles = process_image(path)
            if angles:
                return render_template('index.html', output_image=output_path, angles=angles)
            else:
                return render_template('index.html', error="No triangle found.")
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
