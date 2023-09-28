import pandas as pd
from flask import Flask, request, jsonify
import tensorflow as tf

app = Flask(__name__)
model = tf.keras.models.load_model('./modelCNN.h5')

@app.route('/api', methods=['POST'])
def predict():
    data = request.get_json()
    input_data = preprocess_data(data)
    
    # Certifique-se de converter os dados para float32 antes de passá-los para o modelo
    result = model.predict(tf.convert_to_tensor(input_data, dtype=tf.float32))
    
    return jsonify({'prediction': result.tolist()})

def preprocess_data(input_data):
    columns = ['x', 'y', 'z']
    df = pd.DataFrame(input_data, columns=columns)
    df['x'] = df['x'].astype('float')
    df['y'] = df['y'].astype('float')
    df['z'] = df['z'].astype('float')
    data = df.to_numpy()
    
    # Redimensionar os dados para o formato esperado pelo modelo
    data = data.reshape(-1, 90, 3)
    
    return data

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=3000)
