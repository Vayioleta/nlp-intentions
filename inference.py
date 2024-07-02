import joblib

# Cargar el modelo y el vectorizador
model = joblib.load('models/model.joblib')
vectorizer = joblib.load('models/vectorizer.joblib')

# Función para predecir la intención de un texto
def predict_intention(text):
    text_vec = vectorizer.transform([text])
    return model.predict(text_vec)[0]

# Ejemplo de uso
print(predict_intention("Quiero hacer cosas"))
print(predict_intention("Gracias"))