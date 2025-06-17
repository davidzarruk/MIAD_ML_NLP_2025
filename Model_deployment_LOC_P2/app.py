# Importar librerias 
from flask import Flask, request
from flask_restx import Api, Resource, fields
import joblib
import pandas as pd
import numpy as np

# Cargar modelos previamente entrenados
modelo = joblib.load("clf_genero.pkl")
vectorizer = joblib.load("vectorizer.pkl")
binarizer = joblib.load("binarizer.pkl")

#Definición API Flask:
app = Flask(__name__)
api = Api(app, version='1.0', title='Genres Prediction API',
          description='Predice el género de las películas a partir de sus características', mask=None, mask=None)

ns = api.namespace('Predict', description='Modelo de clasificación')

# Modelo de entrada para Swagger
input_model = api.model('Input', {
    'title': fields.String(required=True, description='Título de la película'),
    'plot': fields.String(required=True, description='Sinopsis de la película')
})

# Definir el campo de salida con nombre personalizado
resource_fields = api.model('Output', {
    'result': fields.String(description='Géneros predichos')
})

@ns.route('/')
@ns.doc(params={
    'title': 'Título de la película',
    'plot': 'Sinopsis de la película'
})
class GenreClassifier(Resource):
    @ns.marshal_with(resource_fields)
    def get(self):
        title = request.args.get('title')
        plot = request.args.get('plot')

        if not all([title, plot]):
            return {'result': 'Error: Faltan parámetros'}, 400

        texto_completo = title + ' ' + plot
        X_input = vectorizer.transform([texto_completo])
        y_pred = modelo.predict(X_input)
        etiquetas = binarizer.inverse_transform(y_pred)

        return {'result': ', '.join(etiquetas[0]) if etiquetas[0] else 'Sin género detectado'}


# Ejecutar la app
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)