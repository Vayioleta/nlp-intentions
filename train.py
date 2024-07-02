import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Paso 1: Recolección de Datos
data = pd.read_csv('data/text_intentions.csv')  # Asegúrate de tener un CSV con columnas 'text' e 'intention'

# Paso 2: Preprocesamiento de Datos
X = data['text']
y = data['intention']

# Paso 3: Vectorización
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

# Paso 4: División de Datos
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

# Paso 5: Entrenamiento del Modelo
model = MultinomialNB()
model.fit(X_train, y_train)

# Paso 6: Evaluación del Modelo
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Guardar el modelo y el vectorizador
joblib.dump(model, 'models/model.joblib')
joblib.dump(vectorizer, 'models/vectorizer.joblib')
print("Modelo y vectorizador guardados exitosamente.")