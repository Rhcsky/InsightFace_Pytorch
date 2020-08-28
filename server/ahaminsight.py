from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
from checkface import recognize

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
    imagefile = request.files['image']
    filename = BASE_DIR + '/uploads/' + secure_filename(imagefile.filename)
    imagefile.save(filename)
    name, frame = recognize(filename)
    return name

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80)

