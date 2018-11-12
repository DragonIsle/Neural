from flask import Flask

app = Flask(__name__)


@app.route("/consider/letter")
def hello(img):
    return "Hello World!"


if __name__ == "__main__":
    app.run()
