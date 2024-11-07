from flask import Flask,render_template,Response
import cv2
import detection
import json

app=Flask(__name__)
camera=cv2.VideoCapture(0)
score, lat, lng, bounding_box_areas = None, None, None, None

def generate_frames():
    global score, lat, lng, bounding_box_areas
    while True:

        ## read the camera frame ##
        success, frame = camera.read()
        if not success:
            break
        else:
            ## Detection model here ##
            result, score, lat, lng, bounding_box_areas = detection.yolo_detection(frame)

            # Convert the result to bytes
            ret, buffer = cv2.imencode('.jpg', result)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/data')
def data():
    global score, lat, lng, bounding_box_areas
    return json.dumps({'score': score, 'lat': lat, 'lng': lng, 'bounding_box_areas':bounding_box_areas})

if __name__=="__main__":
    app.run(debug=True)

