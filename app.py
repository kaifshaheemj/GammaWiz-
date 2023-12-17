from flask import Flask, render_template, request
import joblib
app = Flask(__name__)

model=joblib.load('B:\ML\Gamma_Rays\ML_CA3-20231017T145348Z-001\ML_CA3\models\knn_model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        user_input = []
        for feature in ['fLength', 'fWidth', 'fSize', 'fConc', 'fConc1', 'fAsym', 'fM3Long', 'fM3Trans', 'fAlpha', 'fDist']:
            user_input.append(float(request.form[feature]))
            print(user_input)
        
        prediction = model.predict([user_input])
        print(prediction)
        return render_template('index.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
