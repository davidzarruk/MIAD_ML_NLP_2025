#!/usr/bin/env python
# coding: utf-8

# In[121]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import LinearRegression
import joblib


# In[122]:


dataTrain = pd.read_csv('dataTrain_Spotify.csv')
dataTest = pd.read_csv('dataTest_Spotify.csv')


# In[123]:


dataTrain.head()


# In[124]:


dataTrain.describe()


# # Prepocesamiento de datos de entrenamiento con .fit_transform() y de validación con .transform()
# 

# In[125]:


#variable mode en train
data_m = dataTrain[['mode']]
encoder = OneHotEncoder(dtype=np.int8)
data_mode = encoder.fit_transform(data_m).toarray()
dataTrain.drop(['mode'], axis=1, inplace=True)
dataTrain = dataTrain.join(pd.DataFrame(data_mode, columns=['menor', 'mayor']))


# In[126]:


#variable mode en test
datat_m = dataTest[['mode']]
datat_mode = encoder.transform(datat_m).toarray()
dataTest.drop(['mode'], axis=1, inplace=True)
dataTest = dataTest.join(pd.DataFrame(datat_mode, columns=['menor', 'mayor']))


# In[127]:


#variable time_signature en train
data_ts = dataTrain[['time_signature']]
data_tsig = encoder.fit_transform(data_ts).toarray()
indice_time = [f'time_signature_{i}' for i in range(5)]
dataTrain.drop(['time_signature'], axis=1, inplace=True)
dataTrain = dataTrain.join(pd.DataFrame(data_tsig, columns=indice_time))


# In[128]:


#variable time_signature en test
datat_ts = dataTest[['time_signature']]
datat_tsig = encoder.transform(datat_ts).toarray()
indice_time = [f'time_signature_{i}' for i in range(5)]
dataTest.drop(['time_signature'], axis=1, inplace=True)
dataTest = dataTest.join(pd.DataFrame(datat_tsig, columns=indice_time))


# In[129]:


#variable key en train
data_k = dataTrain[['key']]
data_key = encoder.fit_transform(data_k).toarray()
indice_key = [f'key_{i}' for i in range(12)]
dataTrain.drop(['key'], axis=1, inplace=True)
dataTrain = dataTrain.join(pd.DataFrame(data_key, columns=indice_key))


# In[130]:


#variable key en test
datat_k = dataTest[['key']]
datat_key = encoder.fit_transform(datat_k).toarray()
indice_key = [f'key_{i}' for i in range(12)]
dataTest.drop(['key'], axis=1, inplace=True)
dataTest = dataTest.join(pd.DataFrame(datat_key, columns=indice_key))


# In[131]:


dataTrain.head()


# In[132]:


dataTest.head()


# In[133]:


# Excluir columnas no numéricas restantes
non_numeric_columns = ['Unnamed: 0','track_id', 'album_name', 'track_name','track_name','artists','track_genre'] # Ejemplo de columnas no numéricas
dataTrain.drop(non_numeric_columns, axis=1, inplace=True)
dataTest.drop(non_numeric_columns, axis=1, inplace=True)


# In[134]:


dataTrain.head()


# In[135]:


dataTest.head()


# In[136]:


# Dividir los datos en variables predictoras (X) y variable objetivo (y) para el conjunto de entrenamiento
X_train = dataTrain.drop(['popularity'], axis=1)
y_train = dataTrain['popularity']
X_test = dataTest


# In[137]:


# Crear y entrenar el modelo de regresión lineal
model = LinearRegression()
model.fit(X_train, y_train)
# Realizar predicciones en el conjunto de prueba
y_test_pred = model.predict(X_test)
# Calcular el MSE
# Calcular el MSE usando una parte de y_train para simular y_test
mse = mean_squared_error(y_train[:len(y_test_pred)], y_test_pred)

# Calcular el RMSE
rmse = np.sqrt(mse)

print(f"El error cuadrático medio (MSE) es: {mse}")
print(f"La raíz del error cuadrático medio (RMSE) es: {rmse}")


# In[138]:


# Exportar modelo a archivo binario .pkl
joblib.dump(model, 'popularidadrl.pkl', compress=3)


# In[139]:


# Importación librerías
from flask import Flask
from flask_restx import Api, Resource, fields


# In[ ]:


from flask import Flask, request, render_template_string
from flask_restx import Api, Resource, fields, reqparse
import joblib
import numpy as np

# Cargar modelo
model = joblib.load('popularidadrl.pkl')

# Configurar Flask y API
app = Flask(__name__)
api = Api(app, version='1.0', title='API de Popularidad', description='Predicción con regresión lineal')
ns = api.namespace('predict', description='Predicción API')

# Definir parser de la API (por si también quieres usar /predict/)
parser = reqparse.RequestParser()
parser.add_argument('duration_ms', type=int, required=True)
parser.add_argument('explicit', type=bool, required=True)
parser.add_argument('danceability', type=float, required=True)
parser.add_argument('energy', type=float, required=True)
parser.add_argument('loudness', type=float, required=True)
parser.add_argument('speechiness', type=float, required=True)
parser.add_argument('acousticness', type=float, required=True)
parser.add_argument('instrumentalness', type=float, required=True)
parser.add_argument('liveness', type=float, required=True)
parser.add_argument('valence', type=float, required=True)
parser.add_argument('tempo', type=float, required=True)
parser.add_argument('menor', type=int, required=True)
parser.add_argument('mayor', type=int, required=True)
parser.add_argument('time_signature_0', type=int, required=True)
parser.add_argument('time_signature_1', type=int, required=True)
parser.add_argument('time_signature_2', type=int, required=True)
parser.add_argument('time_signature_3', type=int, required=True)
parser.add_argument('time_signature_4', type=int, required=True)
parser.add_argument('key_0', type=int, required=True)
parser.add_argument('key_1', type=int, required=True)
parser.add_argument('key_2', type=int, required=True)
parser.add_argument('key_3', type=int, required=True)
parser.add_argument('key_4', type=int, required=True)
parser.add_argument('key_5', type=int, required=True)
parser.add_argument('key_6', type=int, required=True)
parser.add_argument('key_7', type=int, required=True)
parser.add_argument('key_8', type=int, required=True)
parser.add_argument('key_9', type=int, required=True)
parser.add_argument('key_10', type=int, required=True)
parser.add_argument('key_11', type=int, required=True)


resource_fields = api.model('Prediction', {
    'result': fields.Float,
})

# API por parámetros GET
@ns.route('/')
class PredictApi(Resource):
    @ns.doc(parser=parser)
    @ns.marshal_with(resource_fields)
    def get(self):
        args = parser.parse_args()
        input_data = np.array([[args['duration_ms'], args['explicit'], args['danceability'], args['energy'], args['loudness'], args['speechiness']
                              , args['acousticness'], args['instrumentalness'], args['liveness'], args['valence'], args['tempo'], args['menor']
                              , args['mayor'], args['time_signature_0'], args['time_signature_1'], args['time_signature_2'], args['time_signature_3']
                              , args['time_signature_4'], args['key_0'], args['key_1'], args['key_2'], args['key_3'], args['key_4'], args['key_5']
                              , args['key_6'], args['key_7'], args['key_8'], args['key_9'], args['key_10'], args['key_11']]])
        prediction = model.predict(input_data)[0]
        return {'result': prediction}

# Página con formulario
@app.route('/', methods=['GET'])
def home():
    return render_template_string('''
    <h2>Predicción de Popularidad</h2>
    <form action="/predict_form" method="post">
        <label>Duración (ms):</label><br>
        <input type="number" name="duration_ms" required><br><br>

        <label>Explícito (0 o 1):</label><br>
        <input type="number" name="explicit" min="0" max="1" required><br><br>

        <label>Danceability (0.0 - 1.0):</label><br>
        <input type="number" name="danceability" step="0.01" min="0" max="1" required><br><br>

        <label>Energy (0.0 - 1.0):</label><br>
        <input type="number" name="energy" step="0.01" min="0" max="1" required><br><br>

        <label>Loudness:</label><br>
        <input type="number" name="loudness" step="0.01" required><br><br>

        <label>Speechiness:</label><br>
        <input type="number" name="speechiness" step="0.01" required><br><br>

        <label>Acousticness:</label><br>
        <input type="number" name="acousticness" step="0.01" required><br><br>

        <label>Instrumentalness:</label><br>
        <input type="number" name="instrumentalness" step="0.01" required><br><br>

        <label>Valence:</label><br>
        <input type="number" name="liveness" step="0.01" required><br><br>

        <label>Valence:</label><br>
        <input type="number" name="valence" step="0.01" required><br><br>

        <label>Tempo:</label><br>
        <input type="number" name="tempo" step="0.01" required><br><br>

        <label>Menor (0 o 1):</label><br>
        <input type="number" name="menor" min="0" max="1" required><br><br>

        <label>Mayor (0 o 1):</label><br>
        <input type="number" name="mayor" min="0" max="1" required><br><br>

        <h4>Time Signature (selecciona solo uno):</h4>
        {% for i in range(5) %}
            <label>Time Signature {{ i }}:</label>
            <input type="number" name="time_signature_{{ i }}" min="0" max="1" required><br>
        {% endfor %}
        <br>

        <h4>Key (selecciona solo uno):</h4>
        {% for i in range(12) %}
            <label>Key {{ i }}:</label>
            <input type="number" name="key_{{ i }}" min="0" max="1" required><br>
        {% endfor %}
        <br>

        <input type="submit" value="Predecir">
    </form>
''')


# Ruta que recibe el formulario y muestra resultado
@app.route('/predict_form', methods=['POST'])
def predict_form():
    try:
        duration_ms = int(request.form['duration_ms'])
        explicit = int(request.form['explicit'])
        danceability = float(request.form['danceability'])
        energy = float(request.form['energy'])

        input_data = np.array([[duration_ms, explicit, danceability, energy]])
        prediction = model.predict(input_data)[0]

        return render_template_string('''
            <h2>Resultado de la Predicción</h2>
            <p><strong>Popularidad estimada:</strong> {{ pred }}</p>
            <a href="/">Volver</a>
        ''', pred=round(prediction, 2))

    except Exception as e:
        return f"Error: {e}"

# Ejecutar la app
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)

