from PIL import Image
from io import BytesIO
import numpy as np
from tensorflow.keras.applications.imagenet_utils import decode_predictions


def read_imagefile(file):
    """Read an uploaded image

    Args:
        file:

    Returns:
        a PIL image
    """
    image = Image.open(BytesIO(file))
    return image


def get_model_input_shape(model):
    """Get the resolution of image the model was trained with

    Args:
        model: a tf.keras model

    Returns:
        a tuple containing image resolution (H,W,C)
    """
    return tuple(list(model.input.shape)[1:])


def formatted_prediction(prediction):
    return [{"id": p[0], "class": p[1], "probability": float(p[2])} for p in prediction]


def process_and_predict_image(image, preprocessing_func, model, nb_decode=5):
    """Compute predictions on an image

    Args:
        image: a PIL Image
        preprocessing_func: function to apply an image (np.array) before inference
        model: a tf.keras model
        nb_decode: number of top predictions wanted

    Returns:
        A list of dictionaries containing the top predictions.
            The lenght of the list is given by the nb_decode parameter.
            The format of the dictionaries are 
            {"id": "n01882714", "class": "koala", "probability": 0.90}
    """
    # preprocess
    shape = get_model_input_shape(model)
    image = image.resize(shape[:2])
    image = np.asarray(image)
    preprocessed_image = preprocessing_func(image)
    # predict
    predictions = model.predict(np.array([preprocessed_image]))
    # decode prediction
    predictions = decode_predictions(predictions, nb_decode)[0]
    return formatted_prediction(predictions)
