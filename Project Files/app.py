from flask import Flask, render_template,request,jsonify,url_for,redirect
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from PIL import Image
import numpy as np
import os
import tensorflow as tf


app=Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

model = tf.keras.models.load_model('healthy_vs_rotten.h5', compile=False)

@app.route('/')
def index():
    return render_template("index.html")
@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/predict', methods=['GET','POST'])
def predict_page():
    if request.method =='POST':
        file =request.files.get('image')
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(file_path)
        img=load_img(file_path,target_size=(224,224))
        image_array = np.array(img)
        image_array = np.expand_dims(image_array, axis=0)
        pred=np.argmax(model.predict(image_array),axis=1)
        index=['Apple__Healthy (0)','Apple__Rotten (1)','Banana__Healthy (2)',
               'Banana__Rotten (3)','Bellpepper__Healthy (4)','Bellpepper__Rotten (5)',
               'Carrot__Healthy (6)','Carrot__Rotten (7)', 'Cucumber__Healthy (8)','Cucumber__Rotten (9)','Grape__Healthy (10)',
               'Grape__Rotten (11)','Guava__Healthy (12)','Guava__Rotten (13)',
               'Jujube__Healthy (14)',
               'Jujube__Rotten (15)','Mango__Healthy (16)','Mango__Rotten (17)','Orange__Healthy (18)',
               'Orange__Rotten (19)','Pomegranate__Healthy (20)','Pomegranate__Rotten (21)', 'Potato__Healthy (22)',
               'Potato__Rotten (23)','Strawberry__Healthy (24)', 'Strawberry__Rotten (25)','Tomato__Healthy (26)', 'Tomato__Rotten (27)'
               ]
        prediction = index[int(pred)]
        #confidence = float(np.max(prediction)) *100
        print("prediction")
        filename = file.filename
        return render_template("result.html", result = prediction,filename=filename )
    return render_template("predict.html")

if __name__=='__main__':
    app.run(debug = True,port =2222)