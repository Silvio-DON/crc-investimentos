import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

df = pd.read_csv('notebook/clientes.csv')
X = df.drop('risco', axis=1)
y = df['risco']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', SVC(probability=True, random_state=42,
                class_weight='balanced', C=10,
                gamma='scale', kernel='rbf'))
])

pipeline.fit(X_train, y_train)

joblib.dump(pipeline, 'backend/model/model.pkl')
print('Modelo treinado e salvo com sucesso!')