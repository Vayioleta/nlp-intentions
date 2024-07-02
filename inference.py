import joblib

# Cargar el modelo y el vectorizador
model = joblib.load('models/model.joblib')
vectorizer = joblib.load('models/vectorizer.joblib')

# Función para predecir la intención de un texto
def predict_intention(text):
    text_vec = vectorizer.transform([text])
    return model.predict(text_vec)[0]

# Ejemplo de uso
print("Quiero informacion ->",predict_intention("Quiero hacer cosas"))
print("Gracias ->",predict_intention("Gracias"))