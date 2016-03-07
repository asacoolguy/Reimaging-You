#!/usr/local/bin/python

# imports for basic flask web stuff
from flask import Flask, render_template, send_file, url_for, request,  redirect
from flask import send_from_directory, jsonify
import StringIO
from os import path
import base64
from werkzeug import secure_filename

# imports for word cloud and image manipulation
from wordcloud import WordCloud
from PIL import Image
import numpy as np
from pylab import *
from skimage.filters import threshold_otsu
from scipy.misc import toimage

# setting up some parameters
UPLOAD_FOLDER = './static/images/originals/'
MASK_FOLDER = './static/images/masks/'
OUTPUT_FOLDER = './static/images/outputs/'
ALLOWED_EXTENSIONS = set(['png','jpg','jpeg'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MASK_FOLDER'] = MASK_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
d = path.dirname(__file__)

# --- main function for the app ---
@app.route('/', methods = ['GET', 'POST'])
def main():
    #handles file upload
    uploaded_filename = "";

    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(path.join(app.config['UPLOAD_FOLDER'], filename))
            uploaded_filename = path.join(app.config['UPLOAD_FOLDER'], filename)

    if uploaded_filename != "":
        return render_template(
            'index.html',
            uploaded_portrait = uploaded_filename)
    else:
        return render_template(
            'index.html')


# --- helper function tha checks if the uploaded file has allowed extension ---
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1] in ALLOWED_EXTENSIONS

# --- main function for generating typographic portraits --- 
@app.route("/generate")
def generate():
    input_text = request.args.get('text')
    input_img = request.args.get('image')
    filename = path.basename(input_img)
    
    original = np.array(Image.open(input_img).convert('L'))
    thresh = threshold_otsu(original)
    binary = original > thresh
    threshold = toimage(binary)
    threshold.save(path.join(app.config['MASK_FOLDER'], filename))

    mask = np.array(Image.open(path.join(app.config['MASK_FOLDER'], filename)))

    wordcloud = WordCloud(
        background_color="white", 
        max_words=3000, 
        mask=mask,
        min_font_size = 1)
    #wordcloud.generate_from_frequencies(words)
    wordcloud.generate(input_text)

    wordcloud.to_file(path.join(app.config['OUTPUT_FOLDER'], filename))
    result = wordcloud.to_image()
    return serveImg(result)

def serveImg(img):
    img_io = StringIO.StringIO()
    img.save(img_io,"JPEG")
    im_data = img_io.getvalue()
    data_url = 'data:image/jpg;base64,' + base64.b64encode(im_data)
    return jsonify(image = data_url)


if __name__ == "__main__":
    app.run(debug=True)
