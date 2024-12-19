from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import json
from flask_cors import CORS, cross_origin

app = Flask(__name__)

# Aktifkan CORS untuk aplikasi Flask
CORS(app, resources={r"/predict": {"origins": "http://127.0.0.1:5000"}})

# Load models and preprocessors
with open('model/reduced_recommendation_model.pkl', 'rb') as file:
    knn = pickle.load(file)

with open('model/scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

with open('model/vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

file_path = 'model/reduced_dataset.csv'
recipe_df = pd.read_csv(file_path)

@app.route('/')
def home():
    return render_template('index.html')  # Halaman utama

@app.route('/kalkulator')
def kalkulator():
    return render_template('kalkulator.html')  # Halaman kalkulator

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Ambil data dari request JSON
        data = request.json
        gender = data['gender']
        umur = int(data['age'])
        list_bahan = data['ingredients']

        # Preprocessing input
        gender_encoded = 1 if gender == 'male' else 0

        # Function to Recommend Recipes
        #gizi sesuai PERATURAN MENTERI KESEHATAN REPUBLIK INDONESIA
        #               protein, karbo, fat,  fiber
        data_nutrisi = [[13.3,   83.3,  18.3, 7.8], #kategori 0
                      [16.7,   100,   21.7, 9.3], #kategori 1
                      [18.3,   93.3,  21.7, 9  ]] #kategori 2
    
        #mengategorikan berdasarkan umur dan gender
        #dikhusus untuk anak sd range 7-12
        if umur < 7:
            print("maaf umur tidak termasuk anak sd")
        elif umur < 10:
            kategori = 0
        elif umur < 13:
            if gender == 1: #1 untuk laki laki dan 0 untuk perempuan
                kategori = 1
            else:
                kategori = 2
        else:
            print("maaf umur tidak termasuk anak sd") 
    
    #mengisi input feature berdasarkan kategori
        input_features = []
        for i in range(0,4):
            input_features.append(data_nutrisi[kategori][i])
        input_features.append(list_bahan)
    
    # merekomendasikan resep
        input_features_scaled = scaler.transform([input_features[:4]])
        input_ingredients_transformed = vectorizer.transform([input_features[4]])
        input_combined = np.hstack([input_features_scaled, input_ingredients_transformed.toarray()])
        distances, indices = knn.kneighbors(input_combined)
        recommendations = recipe_df.iloc[indices[0]]

        prediction =  recommendations[['recipe_name', 'ingredients_list', 'image_url']]    
        import json

        # Convert DataFrame to JSON
        json_data = prediction.to_json(orient='records')
        # Return the response
        return jsonify({'prediction': json.loads(json_data)})
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
