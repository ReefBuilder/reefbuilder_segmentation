import logging.config
import json
import reefbuilder_segmentation.config as cfg
import os

logger = logging.getLogger(cfg.logger_name)


def setup_logging():
    config_file = cfg.log_file_path
    with open(config_file) as f:
        log_config = json.load(f)
    log_file_path = log_config["handlers"]["file"]["filename"]
    if not os.path.exists(log_file_path):
        os.makedirs(os.path.dirname(os.path.expanduser(log_file_path)), exist_ok=True)
    base_name = os.path.splitext(os.path.basename(log_file_path))[0]
    print(
        f"Log file can be found here: {os.path.abspath(os.path.dirname(log_file_path))}. File name will start with {base_name}"
    )
    logging.config.dictConfig(log_config)


def main():
    setup_logging()
    logging.info("---*---")
    logging.info("Logger has been setup. Logging now...")


main()
