
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

    def pred_tc(var):
        data = np.array(var)
        test=data.reshape((-1,1))
        poly = PolynomialFeatures(degree=4)
        test_t= poly.fit_transform(test)
        prediction = model_TC.predict(test_t)
        print("prediction",prediction)
        print(type(prediction))
        return prediction
    def pred_hum(var):
        data = np.array(var)
        test=data.reshape((-1,1))
        poly = PolynomialFeatures(degree=3)
        test_t= poly.fit_transform(test)
        prediction = model_HUM.predict(test_t)
        print("prediction",prediction)
        print(type(prediction))
        return prediction


    var = request.form.get("value")
    print("data",var)
    print(type(var))
    p_tc = pred_tc(var)
    p_hum = pred_hum(var)


    return render_template('home.html', prediction_text_tc="The predicted value of Tempreture is {}".format(p_tc),prediction_text_hum="The predicted value of Humidity is {}".format(p_hum))


    
if __name__ == "__main__":
    app.run(debug=True)