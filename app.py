
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.preprocessing import PolynomialFeatures

app = Flask(__name__)
#model_TC = pickle.load(open('C:\\Users\\HP\\Downloads\\DataScieneceStudyMettarial\\FASAL_interview\\Assignment_2\\ML_flaskCode\\TC_forecast.pkl','rb'))
#model_HUM = pickle.load(open('C:\\Users\\HP\\Downloads\\DataScieneceStudyMettarial\\FASAL_interview\\Assignment_2\\ML_flaskCode\\HUM_forecast.pkl','rb'))

model_TC = pickle.load(open('TC_forecast.pkl','rb'))
model_HUM = pickle.load(open('HUM_forecast.pkl','rb'))


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_TC',methods=['POST'])
def predict_TC():

    var = request.form.get("value")
    print("data",var)
    print(type(var))
    

    data = np.array(var)
    test=data.reshape((-1,1))
    poly = PolynomialFeatures(degree=4)
    test_t= poly.fit_transform(test)
    prediction = model_TC.predict(test_t)
    print("prediction",prediction)
    print(type(prediction))

    return render_template('home.html', prediction_text="The predicted value of Tempreture is {}".format(prediction))

@app.route('/predict_HUM',methods=['POST'])
def predict_HUM():

    var = request.form.get("value1")
    print("data",var)
    print(type(var))
    

    data = np.array(var)
    test=data.reshape((-1,1))
    poly = PolynomialFeatures(degree=3)
    test_t= poly.fit_transform(test)
    prediction = model_HUM.predict(test_t)
    print("prediction",prediction)
    print(type(prediction))

    return render_template('home.html', prediction_text_hum="The predicted value  of Humidity is {}".format(prediction))

if __name__ == "__main__":
    app.run(debug=True)