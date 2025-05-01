import logging
import sys
from importlib import resources

import jsonargparse
from buoy import config
from buoy.main import main


def cli(args=None):
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(
        format=log_format,
        level=logging.INFO,
        stream=sys.stdout,
    )
    logging.getLogger("bilby").setLevel(logging.WARNING)

    default_config = resources.files(config).joinpath("config.yaml")
    parser = jsonargparse.ArgumentParser(default_config_files=[default_config])
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
    args = parser.parse_args(args)
    args.pop("config")
    args = parser.instantiate_classes(args)

    main(**vars(args))


if __name__ == "__main__":
    cli()
