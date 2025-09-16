import pytest
from model_simon.prueba_despliegue import predict_phishing_proba

def test_predict_phishing_proba_range():
    result = predict_phishing_proba("http://example.com")
    assert isinstance(result, float)
    assert 0.0 <= result <= 1.0
