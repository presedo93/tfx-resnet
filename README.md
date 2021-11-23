# TFX-Resnet50

This project is an example on how to serve and use (via requests) a Machine Learning model using Tensorflow Serving.

## Get the model

The model used for this project is a ResNet 50 for classification. It is one of the models available in the TensorFlow Hub: https://tfhub.dev/tensorflow/resnet_50/classification/1

The easiest way to get it ready to be deployed is to make use of the script inside this repo:

    ./getmodel.sh

This script gets the **tar** file, extracts its content under `assets/resnet/1` and removes the file.

## Deploy everything!

Both, the tensorflow serving and the "client" are inside Docker containers. On top of that, the repo has a `docker-compose.yml` to make it more straight. To launch the two containers just run:

    docker-compose up --build --attach resnet-api

Attaching to the resnet-api container allows to see the output of the request.

## Brief explanation

Inside `cli.py`, the user can set the source as an image stored in the project folder or as an URL. The script checks if the **source** argument is an image or an URL using *regex* expressions. And the API inside `api.resnet.py` can work with both formats thanks to the `infer_from_url` and `infer_from_img` methods.

Inside the API, there are two more methods (used in both cases). `warmup` which makes some requests to allow the server to load the model's weights, etc. And `infer` which does **num_requests** inferences and countes the elapsed time for all of them. `infer` also stores the prediction of each of the requests so the user can see which was the output in each requests.

The main benefict of storing all the predictions is that all the methods (except `warmup`) support the *target* parameter and support calculating the **accuracy** between all the requests. What is the main purpose of this? The user can "evaluate" the model that has been deployed in production. How? Let's imagine that the user wants to check if model has difficulties with and image of a dog that look alike a cat. Just running the same image and checking that accuracy allows to see the performance of the model in production.

## Git hooks

There are four hooks in this repo that allow to format the code and check for inconsistencies. So, please, in case of making a commit, remember to first install them:

    pre-commit install

## Further steps

It could be really interesting to test another type of serving. For example, using **FastAPI** as the backend for our application. It could make easier to support more networks at the same time:

    from enum import Enum

    from fastapi import FastAPI


    class ModelName(str, Enum):
        alexnet = "alexnet"
        resnet = "resnet"
        lenet = "lenet"


    app = FastAPI()


    @app.get("/models/{model_name}")
    async def get_model(model_name: ModelName):
        if model_name == ModelName.alexnet:
            return {"model_name": model_name, "message": "Deep Learning FTW!"}

        if model_name.value == "lenet":
            return {"model_name": model_name, "message": "LeCNN all the images"}

        return {"model_name": model_name, "message": "Have some residuals"}
