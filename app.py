from flask import Flask, render_template, url_for,request,redirect
import pickle,numpy as np
import sklearn
import os
app=Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/crop_recommender', methods=['GET','POST'])
def crop():
    nitrogen = None
    potassium = None
    phosphorous = None
    ph = None
    temperature=None
    humidity=None
    rainfall = None
    
    if request.method == 'POST':
        nitrogen = request.form.get('nitrogen')
        potassium =  request.form.get('potassium')
        phosphorous =request.form.get('phosphorous')
        temperature = request.form.get('temp')
        humidity = request.form.get('humidity')
        ph = request.form.get('ph')
        rainfall = request.form.get('rainfall')
        print(f"Received form data (GET): Nitrogen={nitrogen}, Potassium={potassium}, Phosphorous={phosphorous},temp={temperature},pH={ph}, humidity={humidity}, Rainfall={rainfall}")
        res=predict(nitrogen,potassium,phosphorous,temperature,humidity,ph,rainfall)
        print(res)
        return redirect(f'/crop_recommender/result?res={res}')
    return render_template('recommender.html')

    
@app.route('/crop_recommender/result')
def cropresult():
    res = request.args.get('res')
    return render_template('resultcrop.html',ress=res)

def predict(n,k,p,temp,humid,ph,rf):
    model_path = os.path.join('model', 'RF.pkl')
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    data = np.array([[n,p, k, temp, humid, ph, rf]])
    prediction = model.predict(data)
    a=prediction[0]
    return a
    


if __name__=='__main__':
    app.run(debug=True)
