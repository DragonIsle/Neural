import base64
import io

from PIL import Image
from flask import Flask, render_template, request

from mini_network import check_image

app = Flask(__name__)


@app.route("/")
@app.route("/index")
def index():
    return render_template('index.html')


@app.route("/consider/letter", methods=['POST'])
def consider():
    img = Image.open(io.BytesIO(base64.b64decode(request.form.get('imgBase64').split(',')[1])))
    img = img.resize((28, 28), Image.ANTIALIAS)
    return str(check_image(img))


if __name__ == '__main__':
    app.run()
