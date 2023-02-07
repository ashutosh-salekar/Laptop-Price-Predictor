from flask import Flask, render_template, request
from joblib import load
import pickle
import pandas as pd
import numpy as np


app = Flask(__name__)

@app.route('/',methods=['POST','GET'])
def hello():
    data_dev = pickle.load(open('Train_data.pkl','rb'))

    company = sorted(data_dev['Company'].unique().tolist())
    typename = sorted(data_dev['TypeName'].unique().tolist())
    cpu = sorted(data_dev['Cpu'].unique().tolist())
    ram = sorted(data_dev['Ram'].unique().tolist())
    gpu = sorted(data_dev['Gpu'].unique().tolist())
    os = sorted(data_dev['OpSys'].unique().tolist())
    resolution = pickle.load(open('Resolution_catg.pkl','rb'))

    fhd = ips = touchscreen = qhd = fourk_uhd = retina = ['Yes', 'No']

    return render_template('home.html', company = company, typename = typename, cpu = cpu, ram = ram, 
    gpu = gpu, os = os, fhd = fhd, ips = ips, touchscreen = touchscreen, qhd = qhd, fourk_uhd = fourk_uhd,
    retina = retina, resolution = resolution)


@app.route('/Predict_Price',methods=['POST','GET'])
def predict():
    pipeline = pickle.load(open('pipeline.pkl','rb'))
    
    feat = request.form.to_dict()
    # print('\n\n FEATURE: ', feat, '\n\n')
    # print('FEATURE:', feat)

    if feat['fhd'] == 'No':
        feat['fhd'] = 0
    else:
        feat['fhd'] = 1

    if feat['ips'] == 'No':
        feat['ips'] = 0
    else:
        feat['ips'] = 1

    if feat['touchscreen'] == 'No':
        feat['touchscreen'] = 0
    else:
        feat['touchscreen'] = 1

    if feat['qhd'] == 'No':
        feat['qhd'] = 0
    else:
        feat['qhd'] = 1

    if feat['fourk_uhd'] == 'No':
        feat['fourk_uhd'] = 0
    else:
        feat['fourk_uhd'] = 1

    if feat['retina'] == 'No':
        feat['retina'] = 0
    else:
        feat['retina'] = 1

    combined_resolution = feat['resolution']
    splited_resolution = combined_resolution.split('x')


    feat = list(feat.values())
    feat[14] = int(splited_resolution[0])
    feat.insert(15, int(splited_resolution[1]))
    # print(feat)

    feat_to_feed = np.array(feat)
    feat_to_feed = feat_to_feed.reshape(1,19)

    result = pipeline.predict(feat_to_feed)
    

    price = 'Rs. ' + str(round(result[0],2))
    return render_template('output.html',result = price)


if __name__ == "__main__":
    app.run(debug=True)