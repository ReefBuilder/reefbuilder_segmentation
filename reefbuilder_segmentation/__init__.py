import logging.config
import json
import reefbuilder_segmentation.config as cfg

logger = logging.getLogger(cfg.logger_name)


def setup_logging():
    config_file = cfg.log_file_path
    with open(config_file) as f:
        log_config = json.load(f)
    logging.config.dictConfig(log_config)


def main():
    setup_logging()
    logging.info("\n---*---")
    logging.info("Logger has been setup. Logging now...")


main()
