"""Predict reports; diagnostic labels may be omitted from the manifest."""

from .evaluate import main

if __name__ == "__main__":
    main(predict_only=True)
