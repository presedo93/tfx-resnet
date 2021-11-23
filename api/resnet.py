import cv2
import json
import requests
import numpy as np

from typing import Dict
from sklearn.metrics import accuracy_score


class ResNetAPI:
    def __init__(self, server_address: str) -> None:
        self.server_address = server_address

    def infer_from_url(
        self, url: str, num_requests: int = 10, warmup: bool = True, target: int = None
    ) -> Dict:
        """Infer the image given in the URL. This method gets the image directly from the URL
        and sends it to the tf serving.

        Args:
            url (str): URL to the image.
            num_requests (int, optional): number of request for the image. Defaults to 10.
            warmup (bool, optional): allow warmup before doing the desired number
            of requests. Defaults to True.
            target (int, optional): if the target is given, the method will calculate
            the accuracy (it is supposed for evaluation purposes). Defaults to None.

        Returns:
            Dict: dictionary with the elapsed time, mode prediction
            and accuracy (if targets given).
        """
        resp = requests.get(url, stream=True)
        resp.raise_for_status()

        img = np.asarray(bytearray(resp.raw.read()), dtype="uint8")
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img_batch = np.expand_dims(img / 255.0, 0).tolist()
        input_request = json.dumps({"instances": img_batch})

        if warmup:
            self.warm_up(input_request)

        return self.infer(input_request, num_requests, target)

    def infer_from_img(
        self,
        filename: str,
        num_requests: int = 10,
        warmup: bool = True,
        target: int = None,
    ) -> Dict:
        """Infer from a image stored in the root of the repository. This method opens the image
        and sends its data to the tf serving.

        Args:
            filename (str): name of the image.
            num_requests (int, optional): number of requests for the image. Defaults to 10.
            warmup (bool, optional): allow warmup before doing the desired number
            of requests. Defaults to True.
            target (int, optional): if the target is given, the method will calculate
            the accuracy (it is supposed for evaluation purposes). Defaults to None.

        Returns:
            Dict: dictionary with the elapsed time, mode prediction
            and accuracy (if targets given).
        """
        img = cv2.imread(filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img_batch = np.expand_dims(img / 255.0, 0).tolist()
        input_request = json.dumps({"instances": img_batch})

        if warmup:
            self.warm_up(input_request)

        return self.infer(input_request, num_requests, target)

    def warm_up(self, input_request: str, n_warms: int = 3) -> None:
        """Perform the warmup so the server will be ready (weights initialized)
        for the next requests.

        Args:
            input_request (str): the same request that will be done later.
            n_warms (int, optional): number of warm ups. Defaults to 3.
        """
        for _ in range(n_warms):
            response = requests.post(self.server_address, data=input_request)
            response.raise_for_status()

    def infer(self, input_request: str, num_requests: int, target: int = None) -> Dict:
        """Do the inference, calculating the prediction for each request. Getting the
        prediction per request allows to calculate the "mean" accuracy.

        Args:
            input_request (str): JSON request with the image inserted.
            num_requests (int): number of times to do the inference.
            target (int, optional): the true label of the image
            (used for testing purposes). Defaults to None.

        Returns:
            Dict: dictionary with the elapsed time, mode prediction
            and accuracy (if targets given).
        """
        elapsed = 0.0
        prediction = list()
        metrics = dict()

        for _ in range(num_requests):
            response = requests.post(self.server_address, data=input_request)
            response.raise_for_status()
            elapsed += response.elapsed.total_seconds()
            prediction += [response.json()["predictions"][0]]

        metrics["prediction"] = np.argmax(prediction, axis=1)
        metrics["elapsed"] = elapsed / num_requests

        if target is not None:
            metrics["accuracy"] = accuracy_score(
                [target] * num_requests, metrics["prediction"]
            )

        vals, counts = np.unique(metrics["prediction"], return_counts=True)
        metrics["prediction"] = vals[np.argmax(counts)]

        return metrics
