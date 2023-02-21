from flask import Flask, render_template, request
from joblib import load
import pickle
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
    opsys = sorted(data_dev['OpSys'].unique().tolist())
    resolution = pickle.load(open('Resolution_catg.pkl','rb'))

    display_dict = {'fhd':'FHD', 'qhd':'QHD', '4kuhd':'4K UHD', 'retina display':'Retina Display'}

    ips = touchscreen = ['Yes', 'No']

    return render_template('home.html', company = company, typename = typename, cpu = cpu, ram = ram, gpu = gpu, opsys = opsys, display = display_dict.values(), ips = ips, touchscreen = touchscreen, resolution = resolution)


@app.route('/Predict_Price',methods=['POST','GET'])
def predict():

    # We have to provide the features as per below  sequence
    # 'Company' | 'TypeName' | 'Inches' | 'Cpu' | 'Ram' | 'Gpu' | 'OpSys' | 'Weight' | 'IPS' | 'Touchscreen' | 'FHD' | 'QHD' | '4KUHD' | 'Retina Display' | 'Resolution' | 'SSD' | 'HDD' | 'Flash'

    data_dev = pickle.load(open('Train_data.pkl','rb'))
    data_dev_col_sequence = data_dev.columns.to_list()
    train_col_count = len(data_dev_col_sequence) 

    feat = request.form.to_dict()

    display_arr = [0 for i in range(4)]
    display_dict = {'fhd':'FHD', 'qhd':'QHD', '4kuhd':'4K UHD', 'retina display':'Retina Display'}
    display_arr_update_fag = 0
    user_data = []
    for col in data_dev_col_sequence:
        col = col.lower()
        if col in display_dict.keys(): 
            if display_arr_update_fag == 0:
                idx = list(display_dict.values()).index(feat['display'])
                display_arr[idx] = 1
                user_data = user_data + display_arr
                display_arr_update_fag = 1        

        elif col in ['ips', 'touchscreen']:
            if feat[col] == 'Yes':
                user_data.append(1)
            else:
                user_data.append(0)

        else:
            user_data.append(feat[col])           

    # print('USER DATA',user_data)

    feat_to_feed = np.array(user_data)
    feat_to_feed = feat_to_feed.reshape(1,train_col_count)

    pipeline = pickle.load(open('pipeline.pkl','rb'))
    result = pipeline.predict(feat_to_feed)
    
    price = 'Rs. ' + str(round(result[0],2))
    return render_template('output.html',result = price)


if __name__ == "__main__":
    app.run(debug=True)