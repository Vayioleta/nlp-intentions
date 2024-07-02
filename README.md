# Text Intention Classifier

Este proyecto implementa un clasificador de intenciones de texto utilizando técnicas de Procesamiento de Lenguaje Natural (NLP). El modelo es capaz de categorizar textos en diferentes intenciones predefinidas, tales como solicitudes de información, quejas, elogios, consultas técnicas, etc.

## Descripción del Proyecto

El objetivo de este proyecto es entrenar un modelo de machine learning que pueda clasificar automáticamente las intenciones de textos en diferentes categorías. Esto puede ser útil en aplicaciones como el soporte al cliente, donde es necesario identificar rápidamente la intención de los mensajes de los usuarios.

### Características

- **Vectorización de Texto**: Utiliza `TfidfVectorizer` para convertir textos en representaciones numéricas.
- **Modelo de Machine Learning**: Utiliza el algoritmo de `MultinomialNB` para la clasificación de textos.
- **Evaluación del Modelo**: Mide la precisión, cobertura y F1-score del modelo en un conjunto de datos de prueba.
- **Persistencia del Modelo**: Guarda el modelo entrenado y el vectorizador en archivos para uso futuro.

## Requisitos

Para ejecutar este proyecto, necesitarás las siguientes librerías de Python:

- `pandas`
- `scikit-learn`
- `nltk` (opcional)
- `joblib`

Puedes instalar todas las librerías necesarias usando:

```sh
pip install pandas scikit-learn nltk joblib
```
