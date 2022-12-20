import os
from flask import Flask, render_template, request
import numpy as np
import pickle

# KMeans Model
model = pickle.load(open('model/svm_fpkk.pkl', 'rb'))
scaler = pickle.load(open('model/scaler_fpkk.pkl', 'rb'))

app = Flask(__name__, template_folder="templates")

@app.route("/predict", methods=['POST'])
def predict():
    age = float(request.form["age"])
    sex = float(request.form["sex"])
    highChol = float(request.form["HighChol"])
    cholCheck = float(request.form["CholCheck"])
    bmi = float(request.form["BMI"])
    smoker = float(request.form["Smoker"])
    heartDiseaseOrAttack = float(request.form["HeartDiseaseorAttack"])
    genHlth = float(request.form["GenHlth"])
    mentHlth = float(request.form["MentHlth"])
    physHlth = float(request.form["PhysHlth"])
    diffWalk = float(request.form["DiffWalk"])

    float_feature = [
        age,
        sex,
        highChol,
        cholCheck,
        bmi,
        smoker,
        heartDiseaseOrAttack,
        genHlth,
        mentHlth,
        physHlth,
        diffWalk
    ]

    final_feature = [np.array(float_feature)]
    scaled_feature = scaler.fit_transform(final_feature)
    prediction = model.predict(scaled_feature)

    output={0:'Pasien Tidak Mengalami Hipertensi',
            1:'Pasien Mengalami Hipertensi'}
    
    return render_template('index.html', prediction_text = 'Berdasarkan nilai dari inputan, maka {} '.format(output[prediction[0]]))
    # return render_template('index.html', prediction_text = 'Fitur {}'.format(prediction))


@app.route("/")
def main():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, port=os.getenv("PORT", default=5000))
