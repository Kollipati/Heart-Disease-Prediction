from flask import Flask,request,render_template,redirect,url_for
import pickle,gzip
import joblib
import numpy as np
model = joblib.load('heart.pkl')

app=Flask(__name__)
@app.route('/')

def home():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
        age=float(request.form.get('age',False))
        sex= float(request.form.get('sex',False))
        cp= float(request.form.get('cp',False))
        trtbps= float(request.form.get('trtbps',False))
        chol=float(request.form.get('chol',False))
        fbs= float(request.form.get('fbs',False))
        restecg= float(request.form.get('restecg',False))
        thalachh= float(request.form.get('thalachh',False))
        exng = float(request.form.get('exng', False))
        oldpeak = float(request.form.get('oldpeak', False))
        slp = float(request.form.get('slp', False))
        caa= float(request.form.get('caa',False))
        thall= float(request.form.get('thall',False))

        arr=np.array([[age,sex,cp,trtbps,chol,fbs,restecg,thalachh,exng,oldpeak,slp,caa,thall]])
        pred=model.predict(arr)
        if pred==0:
            res_Val="No heart problem"
        else:
            res_Val="Heart Problem"

        return render_template('index.html',prediction_text='Patient has {}'.format(res_Val))


if __name__=='__main__':
    app.run(debug=True)

