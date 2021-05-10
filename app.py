from flask import Flask, render_template, request, url_for
import pandas as pd
import numpy as np
import pickle
model = pickle.load(open('LinearRegression.pkl', 'rb'))
car = pd.read_csv('cleaned car.csv')
app = Flask(__name__)
@app.route('/', methods = ['GET'])
def index():
    companies = sorted(car['company'].unique())
    car_models = sorted(car['name'].unique())
    year = sorted(car['year'].unique(), reverse = True)
    fuel_type = car['fuel_type'].unique()
    return render_template('index.html', companies = companies, car_models = car_models,
     year = year, fuel_type = fuel_type)
@app.route('/predict', methods = ['POST'])
def predict():
    company = request.form.get('company')
    car_models = request.form.get('car_model')
    fuel = request.form.get('fuel')
    year = int(request.form.get('year'))
    driven = int(request.form.get('km_driven'))
    prediction = model.predict(pd.DataFrame(columns = ['name', 'company', 'year', 'kms_driven', 'fuel_type'], data = 
    np.array([car_models, company, year, driven, fuel]).reshape(1, 5)))
    print(prediction)
    return str(np.round((prediction)[0], 2))
    

if __name__ == "__main__":
    app.run()
