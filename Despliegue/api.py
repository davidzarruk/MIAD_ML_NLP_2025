#!/usr/bin/python
from flask import Flask
from flask_restx import Api, Resource, fields
from model_genres import predict_genre_proba

app = Flask(__name__)

api = Api(
    app, 
    version='1.0', 
    title='Genres movies Prediction API',
    description='Genres movies Prediction API')

ns = api.namespace('predict', 
     description='Genre Classifier')
   
parser = api.parser()

parser.add_argument(
    'Description', 
    type=str, 
    required=True, 
    help='Movie description to predict the genre', 
    location='args')

resource_fields = api.model('Resource', {
    'result': fields.String,
})

@ns.route('/')
class GenreApi(Resource):

    @api.doc(parser=parser)
    @api.marshal_with(resource_fields)
    def get(self):
        args = parser.parse_args()
        result = predict_genre_proba(args['Description'])
        p1 = {k: float(v) for k, v in result.items()}    
        return {"result": p1}, 200
    
    
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)

