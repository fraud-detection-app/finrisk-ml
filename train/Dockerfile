# The TRAINING image = the app's own image + one small script.
#
# Why start FROM the app image? Training and serving must use the SAME feature code and the SAME
# model code, or a model that scores well in training can score differently in the service. The app
# image already has financial_risk on PYTHONPATH (/app/src) and every library installed, so this
# file adds only the part this repo owns: how the training is run.
#
# Build (the workflow does this; the build context is the repo root):
#   docker build -f train/Dockerfile \
#     --build-arg BASE_IMAGE=<registry>/$(grep -v '^#' train/base_image.txt | head -n 1) \
#     -t finrisk-train:local .
#
# BASE_IMAGE is pinned by DIGEST in train/base_image.txt (a tag can move, a digest cannot).
# Taking a newer app image is a pull request that changes that one line: reviewed, tested by
# the next training run, and easy to revert.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# SageMaker's convention: code lives in /opt/ml/code.
COPY train/sagemaker_train.py /opt/ml/code/sagemaker_train.py

# An absolute path, so it does not matter which WORKDIR the base image uses (/app).
# SageMaker also gets this entrypoint in the training job request (pipeline/mlops.py sets
# ContainerEntrypoint), so the job never depends on the Dockerfile alone.
ENTRYPOINT ["python", "/opt/ml/code/sagemaker_train.py"]
# The app image's CMD (start the web server) must not become arguments of our entrypoint.
CMD []
