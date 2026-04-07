"""
Backend Flask — Classificador de Risco de Cliente
Consultoria de Investimentos
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os

app = Flask(__name__)
CORS(app)

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'model.pkl')

try:
    model = joblib.load(MODEL_PATH)
    print(f'Modelo carregado com sucesso!')
except FileNotFoundError:
    model = None
    print(f'AVISO: Modelo não encontrado em {MODEL_PATH}')


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None
    }), 200


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Modelo não disponível.'}), 503

    data = request.get_json(silent=True)
    if data is None:
        return jsonify({'error': 'Envie dados em formato JSON.'}), 400

    campos = [
        'idade', 'renda_mensal', 'patrimonio', 'historico_credito',
        'tempo_emprego', 'dividas', 'dependentes', 'tem_imovel',
        'tem_investimentos', 'score_credito'
    ]

    for campo in campos:
        if campo not in data:
            return jsonify({'error': f'Campo ausente: {campo}'}), 400

    try:
        features = np.array([[
            float(data['idade']),
            float(data['renda_mensal']),
            float(data['patrimonio']),
            float(data['historico_credito']),
            float(data['tempo_emprego']),
            float(data['dividas']),
            float(data['dependentes']),
            float(data['tem_imovel']),
            float(data['tem_investimentos']),
            float(data['score_credito'])
        ]])

        prediction = int(model.predict(features)[0])
        probability = model.predict_proba(features)[0].tolist()

        labels = {
            0: 'Baixo Risco',
            1: 'Médio Risco',
            2: 'Alto Risco'
        }

        return jsonify({
            'prediction': prediction,
            'resultado': labels[prediction],
            'confianca': round(max(probability) * 100, 1),
            'probabilidades': {
                'baixo': round(probability[0] * 100, 1),
                'medio': round(probability[1] * 100, 1),
                'alto':  round(probability[2] * 100, 1)
            }
        }), 200

    except Exception as e:
        return jsonify({'error': f'Erro ao processar: {str(e)}'}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)