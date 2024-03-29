import os
import numpy as np
from PIL import Image
from tensorflow.keras.applications.nasnet import preprocess_input
from tensorflow.keras.applications import NASNetMobile

from app import utils

# testing with NASNetMobile as it's faster
model = NASNetMobile(
    input_shape=None,
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    pooling=None,
    classes=1000,
)


def test_get_model_input_shape():
    # this is for NASNetMobile only. NASNetLarge is (331, 331, 3)
    assert utils.get_model_input_shape(model) == (224, 224, 3)


def test_process_and_predict_image():
    git_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    
    # test koala
    image = Image.open(os.path.join(git_path, "images", "koala.jpeg"))
    prediction = utils.process_and_predict_image(
        image, preprocessing_func=preprocess_input, model=model
    )
    assert prediction[0]["class"] == "koala" and prediction[0]["probability"] > 0.85

    # test support of black and white format
    image = Image.open(os.path.join(git_path, "images", "dentist_black_white.jpeg"))
    prediction = utils.process_and_predict_image(
        image, preprocessing_func=preprocess_input, model=model
    )
    assert prediction[0]["class"] == "gasmask" and prediction[0]["probability"] > 0.25


def test_formatted_prediction():
    preds = [
        ("n01882714", "koala", np.float32(0.90931666)),
        ("n01883070", "wombat", np.float32(0.0055514555)),
        ("n03372029", "flute", np.float32(0.00041457)),
        ("n02115641", "dingo", np.float32(0.00036802463)),
        ("n02500267", "indri", np.float32(0.00035990827)),
    ]
    assert utils.formatted_prediction(preds) == [
        {"id": "n01882714", "class": "koala", "probability": 0.9093166589736938},
        {"id": "n01883070", "class": "wombat", "probability": 0.005551455542445183},
        {"id": "n03372029", "class": "flute", "probability": 0.000414570007706061},
        {"id": "n02115641", "class": "dingo", "probability": 0.00036802463000640273},
        {"id": "n02500267", "class": "indri", "probability": 0.00035990827018395066},
    ]
