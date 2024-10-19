from flask import Flask, render_template, request
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

app = Flask(__name__)

knn = KNeighborsClassifier(n_neighbors=5)
df = pd.read_csv('data/Healthcare-Diabetes.csv')

if 'Id' in df.columns:
    df = df.drop(columns=['Id'])

X = df.drop('Outcome', axis=1)
y = df['Outcome']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

knn.fit(X_scaled, y)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = [float(x) for x in request.form.values()]
        
        feature_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                           'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        
        data = pd.DataFrame([data], columns=feature_columns)
        
        data_scaled = scaler.transform(data)  # Solo transformar las columnas de características
        
        prediction = knn.predict(data_scaled)[0]  # Get the first prediction
        
        result_text = f'La predicción es: {"Diabetes" if prediction == 1 else "No diabetes"}'
        
        return render_template('index.html', prediction=result_text)
    
    except Exception as e:
        return str(e)

if __name__ == '__main__':
    app.run(debug=True)
