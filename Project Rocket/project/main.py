from flask import Flask, render_template, Response
from view import stream
import cv2
import numpy as np

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

def gen():
    while True:
        frame = stream()
        #frame = cv2.imencode('.jpg', frame)
        yield(b'--frame\r\n' b'content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace;boundary=frame')


if __name__ == '__main__':
    app.run('0.0.0.0', '5002', debug=True)
