import logging.config
import json

logger = logging.getLogger("reefbuilder_segmentation")


def setup_logging():
    config_file = "/Users/ish/github_repos/reefbuilder_segmentation/reefbuilder_segmentation/logging_config.json"
    with open(config_file) as f:
        log_config = json.load(f)
    logging.config.dictConfig(log_config)


def main():
    setup_logging()
    logging.info("\n---*---")
    logging.info("Logger has been setup. Logging now...")


main()
