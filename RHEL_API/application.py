# <--- Deps  --->
import pickle
import sklearn
import flask
import json
from flask import Flask, render_template, request
from joblib import load
import pandas as pd
# <--- Deps  --->

# (Main)     #  {
app = Flask(__name__)
if __name__ == '__main__':
    app.run()
# }

@app.route('/trade')
def trade():
    request.args[""]


if __name__ == '__main__':
    app.run(debug=False)
