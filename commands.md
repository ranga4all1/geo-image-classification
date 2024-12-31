```bash
mkdir data
```

```
conda create -n geo python=3.10 numpy jupyter matplotlib
conda activate geo
pip install tensorflow==2.18 grpcio tensorflow-serving-api==2.18 keras-image-helper flask

```


<!-- ------------------------------- -->


## Convert model from keras model format to tf SavedModel format
``` 
python saved-model.py
```

### Verify

```
(geo) @ranga4all1 ➜ /workspaces/geo-image-classification/code (main) $ tree saved-geo-model/
saved-geo-model/
├── assets
├── fingerprint.pb
├── saved_model.pb
└── variables
    ├── variables.data-00000-of-00001
    └── variables.index

2 directories, 4 files
```

OR
```
ls -lhR saved-geo-model/
```

#### Look at signature
```
saved_model_cli show --dir saved-geo-model --all | less
```

#### OR
```
saved_model_cli show --dir saved-geo-model --tag_set serve --signature_def serving_default
```

## We can run the model (saved-geo-model) with the prebuilt docker image tensorflow/serving:2.18.0:

```
docker run -it --rm \
  -p 8500:8500 \
  -v $(pwd)/saved-geo-model:/models/saved-geo-model/1 \
  -e MODEL_NAME="saved-geo-model" \
  tensorflow/serving:2.18.0
```

### Run gateway service flask application
```
python gateway.py
```
### Test using 
```
python test.py
```
Result:
```
{'buildings': -0.7047825455665588, 'forest': -1.4779146909713745, 'glacier': 2.144595146179199, 'mountain': 1.1838254928588867, 'sea': -0.41370701789855957, 'street': -0.41950517892837524}
```

## Put everything in Pipenv

```
pip install pipenv
pipenv install grpcio keras-image-helper flask gunicorn tensorflow-protobuf==2.11.0
```

## Run everything locally with docker-compose

```
docker build -t saved-geo-model:xception-001 -f image-model.dockerfile .
```

run image:
```
docker run -it --rm \
  -p 8500:8500 \
  saved-geo-model:xception-001
```

```
docker build -t geo-gateway:001 -f image-gateway.dockerfile .
```

```
docker run -it --rm \
  -p 9696:9696 \
  geo-gateway:001
```

### Use docker-compose to run above 2 containers in in one network
```
docker-compose up -d
```

### Test using 
```
python test.py
```
Result:
```
(geo-1) @ranga4all1 ➜ /workspaces/geo-image-classification/code (main) $ python test.py 
{'buildings': -0.7179785370826721, 'forest': -1.896227240562439, 'glacier': 2.369765043258667, 'mountain': 1.5169345140457153, 'sea': -0.49287453293800354, 'street': -0.46033942699432373}
```