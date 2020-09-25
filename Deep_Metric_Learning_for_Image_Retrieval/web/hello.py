from flask import Flask, render_template, request, session, url_for, redirect, send_from_directory
# from flaskext.mysql import MySQL
from werkzeug.utils import secure_filename
from torchvision import transforms
import os
import sys
import urllib
import PIL
import matplotlib.pyplot as plt

from retrieve import retrieve
from retrieve import get_model_result
import contrastive
import triplet
import softtriple
import multisimilarity
import proxyNCA

UPLOAD_PATH = sys.path[0] + '/static/upload'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
MODEL_NUM = 7
app = Flask(__name__)

html = '''
    <!DOCTYPE html>
    <title>Upload File</title>
    <h1>Picture upload</h1>
    <form method=post enctype=multipart/form-data>
         <input type=file name=file>
         <input type=submit value=Upload>
    </form>
    '''

# mysql = MySQL()
# app.config['MYSQL_DATABASE_USER'] = 'staysafe'
# app.config['MYSQL_DATABASE_PASSWORD'] = 'StaySafe-2020'
# app.config['MYSQL_DATABASE_DB'] = 'metric_learning'
# app.config['MYSQL_DATABASE_HOST'] = 'localhost'
# mysql.init_app(app)

app.config['UPLOAD_FOLDER'] = UPLOAD_PATH
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

uploaded_path = 'img/default.png'
result_path = ['img/default.png', 'img/default.png', 'img/default.png', 'img/default.png', 'img/default.png',
               'img/default.png', 'img/default.png', 'img/default.png', 'img/default.png', 'img/default.png', ]
ret_path_eval = [['img/default.png', 'img/default.png', 'img/default.png'],
                ['img/default.png', 'img/default.png', 'img/default.png'],
                ['img/default.png', 'img/default.png', 'img/default.png'],
                ['img/default.png', 'img/default.png', 'img/default.png'],
                ['img/default.png', 'img/default.png', 'img/default.png'],
                ['img/default.png', 'img/default.png', 'img/default.png'],
                ['img/default.png', 'img/default.png', 'img/default.png']]
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

@app.route("/", methods=['GET', 'POST'])
def origin():
    # search()
    return redirect(url_for("search"))


@app.route('/choose', methods=['GET'])
def choose_model():
    if request.values.get("choosedataset") is not None:
        session["dataset"] = request.values.get("choosedataset")
    if request.values.get("choosemodel") is not None:
        session['model'] = request.values.get("choosemodel")
    if request.values.get("choosecarmodel") is not None:
        session['carmodel'] = request.values.get("choosecarmodel")
    return redirect(url_for("search"))


@app.route('/search', methods=['GET', 'POST'])
def search():
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.mkdir(app.config['UPLOAD_FOLDER'])

    model = '0'
    dataset = 'bird'
    if 'model' in session:
        model = session["model"]
    if 'dataset' in session:
        dataset = session['dataset']
        if dataset == 'car':
            model = '7'
            if 'carmodel' in session:
                model = session['carmodel']
    
    error = None
    filename = None
    img_path = None

    # Save files
    if request.method == 'GET':
        if request.values.get("search") is not None:
            # url = http://www.myucdblog.com/wp-content/uploads/2014/07/ucd_campustours_pic2.jpg
            url = request.values.get("search")
            filename = secure_filename(url.split('/')[-1])
            if allowed_file(filename):
                img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                try:
                    urllib.request.urlretrieve(url, img_path)
                except urllib.error.HTTPError:
                    error = "Image can not be downloaded successfully!"
            else:
                error = "Suffix of url must be jpg or png!"
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(img_path)
        else:
            error = "Only jpg or png images are accepted!"

    # Read files
    if error is None and img_path is not None:
        file_url = url_for('uploaded_file', filename=filename)
        try:
            image = PIL.Image.open(img_path)
            results = get_model_result(image, 10, model, dataset)
            print(results)
            return render_template('search.html', uploaded_path=file_url, result_path=results, model = model, dataset = dataset)
        except PIL.UnidentifiedImageError:
            error = "Can not open the file uploaded!"
        
    return render_template('search.html', uploaded_path=uploaded_path, result_path=result_path, error = error, model = model, dataset = dataset)


@app.route('/evaluation',methods=['GET', 'POST'])
def evaluation():
    if request.method == 'GET':
        if request.values.get("askforevainput") is not None:
            file_url = request.values.get("askforevainput")
            img_path = sys.path[0]+ "/static" + file_url
            results = []
            image = PIL.Image.open(img_path)
            for i in range(0,MODEL_NUM):
                ret = get_model_result(image, 3, str(i))
                results.append(ret)
            print(results)
            return render_template('evaluation.html', uploaded_path=file_url, result_path=results)

    return render_template('evaluation.html', uploaded_path=uploaded_path, result_path = ret_path_eval)


@app.route('/saveEval', methods=['GET', 'POST'])
def saveEval():
    if request.method == 'GET':
        eval_ret = ""
        for i in range(0,MODEL_NUM):
            if request.values.get(str(i)) is not None:
                eval_ret += request.values.get(str(i))
                if i == (MODEL_NUM-1):
                    eval_ret += "\n"
                else:
                    eval_ret += ","
        print(eval_ret)

        eval_path = sys.path[0] + "/static"
        if not os.path.exists(eval_path):
            os.mkdir(eval_path)
        with open(eval_path + '/eval_result.csv', 'a+') as f:

            f.write(eval_ret)

    return render_template('submitted.html')


@app.route('/download', methods=['GET', 'POST'])
def download():
    return render_template('download.html')


@app.route('/algorithm', methods=['GET', 'POST'])
def algorithm():
    return render_template('algorithm.html')


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            file_url = url_for('uploaded_file', filename=filename)
            return html + '<br><img src=' + file_url + '>'
    return render_template("upload.html")


@app.route('/upload/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)


# 排除非图片文件
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


# def query(q):
#     # get arguments in request
#     name = request.args.get('name')
#     ret = "Hello, world!"
#     if name is not None:
#         ret = "hello %s!" % name
#         # connect database
#         cursor = mysql.connect().cursor()
#         # execute sql statement
#         cursor.execute("SELECT * from members where name='" + name + "'")
#         # get data in format of sequence
#         data = cursor.fetchone()
#         cursor.close()
#
#         if data is None:
#             ret = "This name is not found!"
#         else:
#             ret = "Hello," + data[0] + "! Email:" + data[1]
#
#     return ret


if __name__ == '__main__':
    app.run(debug=True)
