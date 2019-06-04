from flask import Flask,render_template, request,redirect, url_for,flash, send_from_directory
from werkzeug.utils import secure_filename
import keras
from PIL import Image
import numpy as np
import os
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import model_from_json
from tensorflow.python.keras.backend import set_session
import time

app = Flask(__name__)

#Refresh cache
@app.after_request
def add_header(response):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    response.headers['X-UA-Compatible'] = 'IE=Edge,chrome=1'
    response.headers['Cache-Control'] = 'public, max-age=0'
    return response

# Cifar detection
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)


UPLOAD_FOLDER = './input/'
IMG_URL = './input/image.jpg'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def init():
  json_file = open('./models/model.json','r')
  loaded_model_json = json_file.read()
  json_file.close()
  loaded_model = model_from_json(loaded_model_json)
  #load weights into new model
  loaded_model.load_weights("./models/model.h5")
  #compile and evaluate loaded model
  loaded_model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
  graph = tf.get_default_graph()
  print("Loaded Model from disk")
  return loaded_model,graph
 # global vars for easy reusability
global model, graph,sess,x_train_mean
# initialize these variables
sess = tf.Session()
set_session(sess)
model, graph = init()
x_train_mean = np.load('x_train_mean.npy')

@app.route("/object_recognition",methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            file.filename = 'image.jpg'
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            img_url = './uploads/image.jpg'
            # load image & predict
            # json_file = open('./models/model.json','r')
            # loaded_model_json = json_file.read()
            # json_file.close()
            # model = model_from_json(loaded_model_json)
            # model.load_weights("./models/model.h5")
            print("Loaded Model from disk")
            img = np.array(Image.open('./input/image.jpg').resize((32,32), Image.ANTIALIAS))
            class_names = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
            emojis = [" ̿̿ ̿'̿'\̵͇̿̿\з= ( ▀ ͜͞ʖ▀) =ε/̵͇̿̿/’̿’̿ ̿̿",
                    "( ͡°( ͡° ͜ʖ( ͡° ͜ʖ ͡°)ʖ ͡°) ͡°)",
                    "(づ｡◕‿‿◕｡)づ",
                    "(☞ﾟ∀ﾟ)☞",
                    "(¬‿¬)",
                    "ಠ╭╮ಠ",
                    "༼ʘ̚ل͜ʘ̚༽",
                    "ヾ(⌐■_■)ノ♪",
                    "(｡◕‿◕｡)"]
            with graph.as_default():
                set_session(sess)
                predict = model.predict(np.array([img])/255.0 - x_train_mean)
                class_n = class_names[np.argmax(predict)]
                percent = np.round(np.max(predict)*100,2)
                return render_template('predict_rec.html',img_url=img_url,class_n=class_n,percent=percent,emoji=np.random.choice(emojis))
    if request.method == 'GET':
        return render_template('home_rec.html')
@app.route("/predict")
def predict_image():
    return "Hi"

#Sarcasm Detection
@app.route("/sarcasm_detection")
def sar_detection():
    return render_template('sar_layout.html')

@app.route("/sar_submit",methods=["POST","GET"])
def sar_submit():
    vectorizer = FP.load_obj('vectorizer')
    model = FP.load_obj('model')
    sentence = request.form.get("sentence")
    # Store label for later saving
    global text
    text = sentence
    # If there's no sentence or greater than 150
    if not sentence or len(sentence) > 150:
        return render_template('sar_failure.html')
    vecText = vectorizer.transform([sentence])
    predict = model.predict(vecText)[0]
    confidence = None

    if predict == 1:
        confidence = model.predict_proba(vecText)[0,1]
        predict = "True"
    else:
        confidence = model.predict_proba(vecText)[0,0]
        predict = "False"

    return render_template('sar_successful.html',sentence=sentence,\
                                    confidence=round(confidence*100,2),predict=predict)

@app.route("/sarcasm_detection/guide")
def sarcasmGuide():
    return render_template('guide_sarcasm.html')

@app.route("/sar_label", methods=['POST','GET'])
def sarcasmLabel():
    if request.form.get('true_button') == "True":
        with open('data.txt',"a") as f:
            f.write(text + ' 1\n')

    elif request.form.get('false_button') == "False":
        with open('data.txt',"a") as f:
            f.write(text + ' 0\n')
    else:
        with open('data.txt',"a") as f:
            f.write("Wrong code"+ ' \n')

    return render_template('sar_label.html')

@app.route("/")
def index():
    return render_template("home.html")

if __name__ == '__main__':
  port = int(os.environ.get("PORT", 5000))
  app.run(debug=True, port=port)
