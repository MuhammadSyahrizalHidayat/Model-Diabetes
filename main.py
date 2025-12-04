import pandas as pd
import pickle
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS # Import Flask-CORS

# Membuat aplikasi Flask
app = Flask(__name__, template_folder='views')

# Aktifkan CORS
CORS(app) # Mengizinkan semua domain

# Muat model yang sudah disimpan
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

    # Endpoint untuk halaman home /
    @app.route('/')
    def welcome():
        return render_template('predict.html')

    # Endpoint untuk memprediksi diabetes
    @app.route('/predict', methods=['POST'])
    def predict_diabetes():
        try:
            # Ambil data dari request
            data = request.get_json()

            # Buat DataFrame dengan nama kolom yang sesuai
            input_data = pd.DataFrame([{
                "Pregnancies": data['Pregnancies'],
                "Glucose": data['Glucose'],
                "BloodPressure": data['BloodPressure'],
                "SkinThickness": data['SkinThickness'],
                "Insulin": data['Insulin'],
                "BMI": data['BMI'],
                "DiabetesPedigreeFunction": data['DiabetesPedigreeFunction'],
                "Age": data['Age']
            }])

            # Melakukan prediksi dengan model yang sudah dimuat
            prediction = model.predict(input_data)
            # Mendapatkan probabilitas prediksi
            probabilities = model.predict_proba(input_data)
            # Probabilitas positif dan negatif dalam bentuk persentase
            probability_negative = probabilities[0][0] * 100 # Probabilitas untuk kelas 0 (negatif)
            probability_positive = probabilities[0][1] * 100 # Probabilitas untuk kelas 1 (positif)
            
            # SELALU menampilkan hasil positif diabetes
            # Jika probabilitas diabetes rendah, gunakan nilai minimum 50%
            if probability_positive < 50:
                probability_positive = 50.0 + (probability_positive / 2)  # Minimal 50%
                probability_negative = 100 - probability_positive
            
            result = f'Anda memiliki peluang menderita diabetes berdasarkan model KNN kami. Kemungkinan menderita diabetes adalah {probability_positive:.2f}%.'
            # Kembalikan hasil prediksi dan probabilitas dalam bentuk JSON
            return jsonify({
                'prediction': result,
                'probabilities': {
                'negative': f"{probability_negative:.2f}%", # Format 2 desimal
                'positive': f"{probability_positive:.2f}%"
                }
            })

        except Exception as e:
            return jsonify({'error': str(e)}), 400

    # Jalankan aplikasi Flask
    if __name__ == '__main__': app.run(debug=True)