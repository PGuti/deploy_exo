import json
from fastapi.testclient import TestClient
from PIL import Image
from app.main import app

client = TestClient(app)

def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"msg": "Hello World"}


def test_set_and_get_model():
    client.post("/set_model/", params={"model_name": "NASNetLarge"})
    model_descr = json.loads(client.get("/get_model/").content)
    assert (
        model_descr["model_name"] == "NASNet"
        and model_descr["nb_parameters"] == 88949818
    )
    client.post("/set_model/", params={"model_name": "NASNetMobile"})
    model_descr = json.loads(client.get("/get_model/").content)
    assert (
        model_descr["model_name"] == "NASNet"
        and model_descr["nb_parameters"] == 5326716
    )


def test_prediction():
    with open("./../images/koala.jpeg","rb") as file_image:
        response = client.post("/predict/", files={"file": file_image})
        predictions = json.loads(response.content)
    assert predictions[0]["class"] == "koala" and predictions[0]["probability"] > 0.85
