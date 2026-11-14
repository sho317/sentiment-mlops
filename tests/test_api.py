from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_positive():
    response = client.post("/predict", json={"text": "I love this, it is wonderful!"})
    assert response.status_code == 200
    assert response.json()["label"] == "positive"

def test_predict_negative():
    response = client.post("/predict", json={"text": "I hate this, it is worst!"})
    assert response.status_code == 200
    assert response.json()["label"] == "negative"
	
def test_predict_missing_text():
	response = client.post("/predict", json={})
	assert response.status_code == 422
#	assert response.json()["label"] == "invalid input"
