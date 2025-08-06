from flask import Flask, render_template, request, send_from_directory
import cv2
from ultralytics.engine.predictor import BasePredictor
from ultralytics import YOLO
from flask import request, Response, Flask
import json
from ultralytics import YOLO
from flask import redirect, url_for

yolo_model = BasePredictor(overrides=dict(model='D:\Website Fix Sidang\Website Fix Sidang\best.pt'))  # Initialize YOLOv8 model


app = Flask(__name__)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 1


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/leaf_detection')
def leaf_detection():
    return render_template('leaf_detection.html')

@app.route('/inputleafdisease')
def inputleafdisease():
    return render_template('prediction.html')


@app.route('/data' , methods = ['POST','GET'])
def submit():
    if request.method == 'POST':
        name = request.form['name']
        phone = int(request.form['phone'])
        email = request.form['email']
        subject =request.form['subject']
        message =request.form['message']

        print("Name Of User:",name)
        print("Phone no:",phone)
        print("Email:",email)
        print("subject:",subject)
        print("message:",message)

        return render_template('index.html')
    
    else :
        return render_template('index.html')


@app.route('/predictionstrawberry', methods=['POST'])
def predictionstrawberry():
 

    # Periksa apakah kunci 'image' ada dalam request.files
    if 'image' not in request.files:
        return Response(
            json.dumps({'error': 'No file uploaded'}),
            status=400,
            mimetype='application/json'
        )

    img = request.files["image"]
    img.save('static/img/{}.jpg')
    img_arr = cv2.imread('static/img/{}.jpg')

    img_arr = cv2.resize(img_arr, (300, 300))


    model = YOLO('best.pt')  # load a custom model
    img_to_detect = img_arr
    
    names = model.names
    
    results = model.predict(
        img_to_detect,
        save=True,
        project="static/img", 
        name="inference", 
        exist_ok=True )
    
    predictions = []
    for r in results:
        for c in r.boxes.cls:
            predictions.append(names[int(c)])  # Add predicted class name to predictions list
            print(predictions)
    
    predictions_str = ', '.join(predictions)
    print("Sending predictions to Output.html:", predictions_str)  # Tambahkan pernyataan print sebelum render_template

    return redirect(url_for('output_page', data=predictions_str))

@app.route('/Output.html')
def output_page():
    # Ambil data dari query string jika ada
    predict = request.args.get('data')
    print("Received predictions from predictionstrawberry:", predict)  # Tambahkan pernyataan print untuk memeriksa nilai predictions
    print("ini predict:", predict)
    return render_template('Output.html')


@app.route('/load_img')
def load_img():
    global COUNT
    return send_from_directory('static/img/inference', "image0.jpg")
    

if __name__ == '__main__':
    app.run(debug=True)

