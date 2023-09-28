from flask import Flask, request, jsonify
import tensorflow as tf

app = Flask(__name__)
model = tf.keras.models.load_model('modelCNN.h5')

@app.route('/api', methods=['POST'])
def predict():
    data = request.get_json()
    input_data = preprocess_data(data['input'])
    result = modelCNN.predict(input_data)
    return jsonify({'prediction': result.tolist()})

def preprocess_data(input_data):
    columns = ['x', 'y', 'z']
    df = pd.DataFrame(data, columns=columns)
    df['x'] = df['x'].astype('float')
    df['y'] = df['y'].astype('float')
    df['z'] = df['z'].astype('float')
    data = df.to_numpy()
    data = data.reshape(-1, 90, 3)
    # Implemente o pré-processamento necessário para seus dados
    # Isso pode incluir normalização, redimensionamento, etc.
return input_data

if __name__ == '__main__':
    app.run(debug=True)
