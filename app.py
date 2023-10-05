from flask import Flask, render_template, request
import joblib, warnings
import pandas as pd
warnings.filterwarnings('ignore')

app = Flask(__name__)

pipeline = joblib.load('pipeline.pkl')

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/predict",methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        polio = request.form['polio']
        adult_mortality = request.form['adult_mortality']
        bmi = request.form['bmi']
        diphtheria = request.form['diphtheria']
        hiv_aids = request.form['hiv_aids']
        schooling = request.form['schooling']
        icor = request.form['income_composition_of_resources']
        thinness_5_9_years = request.form['thinness_5_9_years']
        under_5_deaths = request.form['under_5_deaths']
        alcohol = request.form['alcohol']
        data = pd.DataFrame([[polio,adult_mortality,bmi,diphtheria,hiv_aids,schooling,icor,thinness_5_9_years,under_5_deaths,alcohol]],columns=['Polio','Adult Mortality','BMI','Diphtheria','HIV/AIDS','Schooling','Income Composition of Resources','Thinness 5-9 Years','Under-Five Deaths','Alcohol'])
        pred = round(pipeline.predict(data)[0],1)
        return render_template('index.html',prediction_text=f"Your predicted life expectancy is {pred}.")

if __name__ == '__main__':
    app.run(port=8080)