import jsonargparse
from buoy.main import main


def cli(args=None):
    parser = jsonargparse.ArgumentParser()
    parser.add_function_arguments(main, fail_untyped=False, sub_configs=True)
    parser.add_argument("--config", action="config")
    args = parser.parse_args(args)
    args.pop("config")

    main(**vars(args))


if __name__ == "__main__":
    cli()
