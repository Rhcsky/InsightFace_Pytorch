from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
from insightface import recognize, updateFacebank
import time
import random as rand

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
# app.config['MAX_CONTENT_LENGTH'] = 16*1024*1024 # 업로드 용량 제한 단위:byte

@app.route("/")
def home_page():
    return render_template("home.html")


@app.route("/list")
def list_page():
    file_list = os.listdir("./uploads")
    html = """<center><a href="/"> 홈페이지 </a><br><br>"""
    html += f"file list: {file_list}"
    return html


@app.route("/upload",methods=["GET","POST"])
def upload_page():
    if request.method == "GET":
        return render_template("upload.html")
    
    elif request.method == "POST":
        f = request.files["file"]
        file_name = BASE_DIR + '/' + secure_filename(f.filename)
        f.save(file_name)
        name, frame = recognize(file_name)
        os.remove(file_name)
        return render_template("check.html",name=name,frame=frame)


@app.route("/android/inference",methods=["GET","POST"])
def android_test():
    start_time = time.time()
    imagefile = request.files['image']
    filename = BASE_DIR + '/uploads/' + secure_filename(imagefile.filename)
    imagefile.save(filename)
    name, frame = recognize(filename)
    running_time = str(time.time() - start_time)
    name += '/' + running_time
    return name
    
@app.route("/android/uploadface",methods=["GET","POST"])
def android_uploadface():
    start = time.time()
    username = request.form['username']
    imagefile = request.files['image']
    bank_dir = '/home/user/Project/ocr/InsightFace_Pytorch/data/facebank/' + str(username) + '/'

    if os.path.exists(bank_dir) is False:
        os.makedirs(bank_dir)

    filename = bank_dir + str(rand.randint(0,1000)) + secure_filename(imagefile.filename)
    imagefile.save(filename)
    
    running_time = str(time.time() - start)
    name = str(username) + '/' + running_time
    return name

@app.route("/android/updatebank",methods=["GET","POST"])
def android_updatebank():
    # updateFacebank()
    return "SUCCESS"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80, debug=True)