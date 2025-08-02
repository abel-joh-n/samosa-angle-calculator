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

    # Better contrast for thresholding
    _, thresh = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None

    largest_contour = max(contours, key=cv2.contourArea)

    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)

    if len(approx) < 3:
        return None, None

    # If more than 3 points, choose best 3 corner-like points
    # Filter triangles only
    if len(approx) >= 3:
        triangles = [approx[i:i+3] for i in range(len(approx)-2)]
    # Choose triangle with max area
        best_triangle = max(triangles, key=lambda t: cv2.contourArea(np.array(t)))
        approx = np.array(best_triangle)
    else:
        return None, None


    if len(approx) < 3:
        return None, None

    pts = [tuple(pt[0]) for pt in approx]

    # Sort clockwise
    def order_points_clockwise(pts):
        pts = np.array(pts)
        center = np.mean(pts, axis=0)
        angles = np.arctan2(pts[:,1] - center[1], pts[:,0] - center[0])
        pts = pts[np.argsort(angles)]
        return pts.tolist()

    pts = order_points_clockwise(pts)
    a, b, c = pts

    angle_A = calculate_angle(b, a, c)
    angle_B = calculate_angle(a, b, c)
    angle_C = calculate_angle(a, c, b)

    # Draw and label
    cv2.drawContours(image, [np.array([a, b, c])], 0, (0, 255, 0), 2)
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
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)

