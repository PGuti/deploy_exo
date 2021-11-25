# Deployment exercice
The goal of this project is to make 

# Usage
This project is based on Fastapi. It can be run manually or in a docker container.

## Running with uvicorn

Start by installing the python dependencies

```
pip install -r requirements.txt
```

Manually run uvicorn: 
```
cd app
uvicorn main:app --reload
```

The documentation of the project will be available on http://127.0.0.1:8000/docs

## Running with docker

Use docker desktop or:
```
docker build -t myimage .
docker run -d --name mycontainer -p 80:80 myimage
```

NB: the build can take 5 minutes or more because of tensorflow installation.

Once docker running, the api documentation will then be available at http://127.0.0.1/docs

More info on using fast api and docker can be found here: [a fast api and docker](https://fastapi.tiangolo.com/deployment/docker/)

# Tests

You can run the tests using pytest on the test/ folder.
In order to check coverage, you can use:
```
pytest --cov-report term-missing --cov=app tests/
```
Coverage should be close to 100%

# Next steps & to do
## to do
If I have time I would like to:
- optionally output gradcam results for the top n classes.

## Limitations:
There are some stuff missing from this project. 
- There is no authentification. This means that everyone that can query the model can change the model as well, which is not what we would like in a production setup
- The way we create the model today make import/testing super heavy. We may want to rework this part. Maybe with an "init" endpoint?
- There hasn't been any speed / load tests. 
