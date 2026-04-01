import argparse
import gin


def parse_config() -> None:
    """Parse a positional gin config file path and load it."""
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str, help="Path to gin config file.")
    args = parser.parse_args()
    gin.parse_config_file(args.config_path)
