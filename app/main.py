from fastapi import FastAPI, UploadFile, File

# for now we only support NASNEtLarge and NASNetMobile
# Supporting more models could be an improvement to be done.
from tensorflow.keras.applications.nasnet import preprocess_input
from tensorflow.keras.applications import NASNetLarge, NASNetMobile

from utils import read_imagefile, process_and_predict_image

app = FastAPI()

# load model
model = NASNetMobile(
    input_shape=None,
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    pooling=None,
    classes=1000,
)


@app.get("/")
def read_root():
    """Dummy default fast api

    Returns:
        Hello world
    """

    return {"Hello": "World"}


@app.post("/set_model/")
def set_model(model_name):
    """ Change model between NASNetMobile and NASNetLarge.
    In a real production setup, we want to be able to change model versions.
    So we would need to extend this idea, so that we can support any tf.keras model
    (and preprocessing).

    Args:
        model_name (str): the name of the model. Choose between "NASNetMobile" and "NASNetLarge"

    """
    global model
    supported_models = ["NASNetMobile", "NASNetLarge"]
    if model_name not in supported_models:
        return "Unsuported model name {}. Please pick one in {}".format(
            model_name, supported_models
        )
    if model_name == "NasNetLarge":
        model = NASNetLarge(
            input_shape=None,
            include_top=True,
            weights="imagenet",
            input_tensor=None,
            pooling=None,
            classes=1000,
        )
    else:
        model = NASNetMobile(
            input_shape=None,
            include_top=True,
            weights="imagenet",
            input_tensor=None,
            pooling=None,
            classes=1000,
        )
    return "Succesfully changed model to {}".format(model_name)


@app.post("/predict/")
async def prediction(file: UploadFile = File(...)):
    """Predict what is on an image

    Args:

    Returns:
                    The probabilities associated with the inference
    """

    # check that the file can be processed
    supported_extensions = ("jpg", "jpeg", "png")
    extension = file.filename.split(".")[-1]
    if extension not in supported_extensions:
        return "Unsuported extension of file. Please pick one in {}".format(
            supported_extensions
        )
    image = read_imagefile(await file.read())
    prediction = process_and_predict_image(
        image, preprocessing_func=preprocess_input, model=model
    )
    return prediction
