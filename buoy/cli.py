import logging
import sys

import jsonargparse
from main import main


def cli():
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(
        format=log_format,
        level=logging.INFO,
        stream=sys.stdout,
    )

    parser = jsonargparse.ArgumentParser()
    parser.add_function_arguments(main)
    parser.add_argument("--config", action="config")

    parser.link_arguments(
        "inference_params",
        "amplfi_hl_architecture.init_args.num_params",
        compute_fn=lambda x: len(x),
        apply_on="parse",
    )

    parser.link_arguments(
        "inference_params",
        "amplfi_hlv_architecture.init_args.num_params",
        compute_fn=lambda x: len(x),
        apply_on="parse",
    )
    args = parser.parse_args()
    args.pop("config")
    args = parser.instantiate_classes(args)

    main(**vars(args))


if __name__ == "__main__":
    cli()
