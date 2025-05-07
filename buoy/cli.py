import logging
import sys

import jsonargparse
from buoy.main import main


def cli(args=None):
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(
        format=log_format,
        level=logging.INFO,
        stream=sys.stdout,
    )
    logging.getLogger("bilby").setLevel(logging.WARNING)

    parser = jsonargparse.ArgumentParser()
    parser.add_function_arguments(main)
    parser.add_argument("--config", action="config")
    args = parser.parse_args(args)
    args.pop("config")

    main(**vars(args))


if __name__ == "__main__":
    cli()
