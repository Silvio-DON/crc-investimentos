"""
Testes automatizados — Classificador de Risco de Cliente
Execução: pytest backend/tests/test_model.py -v
"""

import pytest
import joblib
import numpy as np
import pandas as pd
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

# Caminhos
MODEL_PATH  = os.path.join(os.path.dirname(__file__), '..', 'model', 'model.pkl')
DATASET_URL = 'https://raw.githubusercontent.com/Silvio-DON/crc-investimentos/master/notebook/clientes.csv'

# Thresholds mínimos de desempenho
MIN_ACCURACY  = 0.75
MIN_PRECISION = 0.75
MIN_RECALL    = 0.75
MIN_F1        = 0.75


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope='module')
def model():
    assert os.path.exists(MODEL_PATH), f"model.pkl não encontrado em {MODEL_PATH}"
    return joblib.load(MODEL_PATH)


@pytest.fixture(scope='module')
def test_data():
    df = pd.read_csv(DATASET_URL)
    X = df.drop('risco', axis=1)
    y = df['risco']
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_test, y_test


@pytest.fixture(scope='module')
def predictions(model, test_data):
    X_test, y_test = test_data
    y_pred = model.predict(X_test)
    return y_test, y_pred


# ── Testes de carregamento ─────────────────────────────────────────────────────

class TestModelLoading:
    def test_model_file_exists(self):
        assert os.path.exists(MODEL_PATH), "model.pkl não encontrado."

    def test_model_loads_successfully(self, model):
        assert model is not None

    def test_model_has_predict_method(self, model):
        assert hasattr(model, 'predict')

    def test_model_has_predict_proba(self, model):
        assert hasattr(model, 'predict_proba')


# ── Testes de predição ─────────────────────────────────────────────────────────

class TestModelPrediction:
    def test_predict_returns_valid_classes(self, model):
        sample = np.array([[35, 15000, 200000, 3, 10, 5000, 1, 1, 1, 850]])
        pred = model.predict(sample)
        assert pred[0] in [0, 1, 2]

    def test_predict_proba_sums_to_one(self, model):
        sample = np.array([[35, 15000, 200000, 3, 10, 5000, 1, 1, 1, 850]])
        proba = model.predict_proba(sample)
        assert abs(sum(proba[0]) - 1.0) < 1e-6

    def test_baixo_risco_cliente_saudavel(self, model):
        # Cliente com ótimo perfil financeiro deve ser Baixo Risco
        sample = np.array([[40, 20000, 500000, 3, 15, 2000, 1, 1, 1, 900]])
        pred = model.predict(sample)
        assert pred[0] == 0, "Cliente saudável deveria ser classificado como Baixo Risco"

    def test_alto_risco_cliente_fragilizado(self, model):
        # Cliente com perfil frágil deve ser Alto Risco
        sample = np.array([[22, 1500, 0, 0, 0, 45000, 4, 0, 0, 320]])
        pred = model.predict(sample)
        assert pred[0] == 2, "Cliente fragilizado deveria ser classificado como Alto Risco"


# ── Testes de desempenho ───────────────────────────────────────────────────────

class TestModelPerformance:
    def test_accuracy_above_threshold(self, predictions):
        y_test, y_pred = predictions
        acc = accuracy_score(y_test, y_pred)
        print(f"\n  Acurácia: {acc:.4f} (mínimo: {MIN_ACCURACY})")
        assert acc >= MIN_ACCURACY, f"Acurácia {acc:.4f} abaixo do mínimo ({MIN_ACCURACY})"

    def test_precision_above_threshold(self, predictions):
        y_test, y_pred = predictions
        prec = precision_score(y_test, y_pred, average='weighted')
        print(f"\n  Precisão: {prec:.4f} (mínimo: {MIN_PRECISION})")
        assert prec >= MIN_PRECISION, f"Precisão {prec:.4f} abaixo do mínimo ({MIN_PRECISION})"

    def test_recall_above_threshold(self, predictions):
        y_test, y_pred = predictions
        rec = recall_score(y_test, y_pred, average='weighted')
        print(f"\n  Recall: {rec:.4f} (mínimo: {MIN_RECALL})")
        assert rec >= MIN_RECALL, f"Recall {rec:.4f} abaixo do mínimo ({MIN_RECALL})"

    def test_f1_above_threshold(self, predictions):
        y_test, y_pred = predictions
        f1 = f1_score(y_test, y_pred, average='weighted')
        print(f"\n  F1-Score: {f1:.4f} (mínimo: {MIN_F1})")
        assert f1 >= MIN_F1, f"F1-Score {f1:.4f} abaixo do mínimo ({MIN_F1})"