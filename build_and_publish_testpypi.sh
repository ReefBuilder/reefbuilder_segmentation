#!/bin/sh

poetry version patch
poetry build
poetry publish -r testpypi
