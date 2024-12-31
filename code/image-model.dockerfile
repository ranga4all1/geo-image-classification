FROM tensorflow/serving:2.18.0

COPY saved-geo-model /models/saved-geo-model/1

ENV MODEL_NAME="saved-geo-model"