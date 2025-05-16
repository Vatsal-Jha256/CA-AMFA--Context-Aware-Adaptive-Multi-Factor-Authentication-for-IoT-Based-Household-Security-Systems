import face_recognition
import numpy as np
import flask
from flask import Flask, request, jsonify
import cv2
import base64
import pickle
import io

app = Flask(__name__)

@app.route('/face/encode', methods=['POST'])
def encode_face():
    # Get image from request
    content = request.json
    image_data = base64.b64decode(content['image'])
    
    # Convert to numpy array
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Convert to RGB (face_recognition expects RGB)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Find faces and get encodings
    face_locations = face_recognition.face_locations(rgb_img)
    
    if not face_locations:
        return jsonify({'success': False, 'error': 'No face detected'})
    
    face_encodings = face_recognition.face_encodings(rgb_img, face_locations)
    
    if not face_encodings:
        return jsonify({'success': False, 'error': 'Could not encode face'})
    
    # Serialize face encoding to send back
    encoding_bytes = pickle.dumps(face_encodings[0])
    encoding_base64 = base64.b64encode(encoding_bytes).decode('utf-8')
    
    return jsonify({
        'success': True,
        'face_encoding': encoding_base64
    })

@app.route('/face/compare', methods=['POST'])
def compare_faces():
    content = request.json
    
    # Get both encodings
    encoding1_bytes = base64.b64decode(content['encoding1'])
    encoding2_bytes = base64.b64decode(content['encoding2'])
    
    # Deserialize
    encoding1 = pickle.loads(encoding1_bytes)
    encoding2 = pickle.loads(encoding2_bytes)
    
    # Get tolerance parameter or use default
    tolerance = content.get('tolerance', 0.6)
    
    # Compare faces
    face_distance = face_recognition.face_distance([encoding1], encoding2)[0]
    match = face_distance < tolerance
    
    return jsonify({
        'success': True,
        'match': bool(match),
        'distance': float(face_distance)
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)