#!/bin/sh

poetry build
poetry publish -r pypi
