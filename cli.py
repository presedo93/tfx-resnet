import re
import argparse

from rich import print
from api.resnet import ResNetAPI


def cli(args: argparse.Namespace) -> None:
    """Method that makes use of the ResNetAPI. Firstly, it checks if
    the source argument is a image or a URL and calls the proper method.

    Args:
        args (argparse.Namespace): arguments passed to the script.
    """
    server_address = f"{args.server_url}:{args.server_port}/{args.server_model}"
    print(f"Using [bold green]{server_address}[/] as the backend!")

    # Create the API
    api = ResNetAPI(server_address)

    # Check if source is an URL
    regex = re.compile(r"^(?:http|ftp)s?://")

    # Check if target in args
    target = None if "target" not in args else args.target

    if re.match(regex, args.source):
        metrics = api.infer_from_url(args.source, target=target)
    else:
        metrics = api.infer_from_img(args.source, target=target)

    print(
        f"Prediction class: [bold magenta]{metrics['prediction']}[/], avg latency: [bold magenta]{metrics['elapsed']:.3f} ms[/]"
    )
    if "accuracy" in metrics:
        print(
            f"[italic]\t Metrics with targets --> [bold]Accuracy[/]: {metrics['accuracy']}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)

    parser.add_argument(
        "--server_url",
        type=str,
        default="http://localhost",
        help="Server URL to connect to.",
    )
    parser.add_argument(
        "--server_port", type=str, default="8501", help="Server port to connect to."
    )
    parser.add_argument(
        "--server_model",
        type=str,
        default="v1/models/resnet:predict",
        help="Model and task to perform.",
    )
    parser.add_argument("--source", type=str, help="Path to the URL / image.")
    parser.add_argument(
        "--target", type=int, help="Optional: Target of the file (used for evaluation)"
    )

    args = parser.parse_args()
    cli(args)
