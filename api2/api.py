#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import re
import nltk
import joblib
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_auc_score


# In[2]:


# Descargar recursos
nltk.download('stopwords')
nltk.download('wordnet')

# Inicializar
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Funciones de limpieza
def clean_text(text):
    text = re.sub("'", "", text)
    text = re.sub("[^a-zA-Z]", " ", text)
    text = ' '.join(text.split())
    text = text.lower()
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
    return text

def remove_stopwords(text):
    return ' '.join([w for w in text.split() if w not in stop_words])

# Cargar datos
dataTraining = pd.read_csv('dataTraining.csv') 
dataTraining['genres'] = dataTraining['genres'].map(lambda x: eval(x))
dataTraining['clean_description'] = dataTraining['plot'].apply(lambda x: clean_text(x))
dataTraining['clean_description'] = dataTraining['clean_description'].apply(lambda x: remove_stopwords(x))

# Binarizar etiquetas
le = MultiLabelBinarizer()
y = le.fit_transform(dataTraining['genres'])

# Vectorizar texto
X = dataTraining['clean_description']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
vectorizer = TfidfVectorizer(max_df=0.28, max_features=24500, ngram_range=(1,3))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Entrenar modelo
clf = OneVsRestClassifier(LogisticRegression(n_jobs=-1))
clf.fit(X_train_tfidf, y_train)

# Evaluar
y_pred_genres = clf.predict_proba(X_test_tfidf)
auc_score = roc_auc_score(y_test, y_pred_genres, average='macro')
print(f"AUC Score: {auc_score:.6f}")

# Guardar modelo y transformadores
joblib.dump(clf, 'modelo_generos.pkl')
joblib.dump(vectorizer, 'vectorizer_tfidf.pkl')
joblib.dump(le, 'label_binarizer.pkl')


# In[ ]:


from flask import Flask
from flask_restx import Api, Resource, reqparse, fields
import re
import joblib
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Descargar recursos
nltk.download('stopwords')
nltk.download('wordnet')

# Inicializar componentes
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Cargar modelo y transformadores
clf = joblib.load('modelo_generos.pkl')
vectorizer = joblib.load('vectorizer_tfidf.pkl')
label_binarizer = joblib.load('label_binarizer.pkl')

# Inicializar Flask 
app = Flask(__name__)
api = Api(app, version='1.0', title='Géneros de Películas API',
          description='Predice géneros de películas a partir de su descripción.')

ns = api.namespace('predict', description='Operaciones de predicción')

# Parser para parámetros 
parser = reqparse.RequestParser()
parser.add_argument('text', type=str, required=True, help='Descripción de la película', location='args')

# Funciones de limpieza
def clean_text(text):
    text = re.sub("'", "", text)
    text = re.sub("[^a-zA-Z]", " ", text)
    text = ' '.join(text.split())
    text = text.lower()
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
    return text

def remove_stopwords(text):
    return ' '.join([w for w in text.split() if w not in stop_words])

# Endpoint con reqparse
@ns.route('/genres')
class GenrePredictor(Resource):
    @ns.expect(parser)
    def get(self):
        """Predice los géneros de una película a partir de su descripción"""
        args = parser.parse_args()
        raw_text = args['text']
        cleaned = clean_text(raw_text)
        cleaned = remove_stopwords(cleaned)
        vectorized = vectorizer.transform([cleaned])
        probs = clf.predict_proba(vectorized)[0]
        threshold = 0.3
        predicted_labels = [label for label, prob in zip(label_binarizer.classes_, probs) if prob >= threshold]

        return {
            'input': raw_text,
            'predicted_genres': predicted_labels,
            'probabilities': dict(zip(label_binarizer.classes_, probs.round(3).tolist()))
        }

# Ejecutar la app
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)

