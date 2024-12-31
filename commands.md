```bash
mkdir data
```

```
conda create -n geo python=3.10 numpy jupyter matplotlib
conda activate geo
pip install tensorflow==2.18
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

## We can run the model (saved-geo-model) with the prebuilt docker image tensorflow/serving:2.7.0:

```
docker run -it --rm \
  -p 8500:8500 \
  -v $(pwd)/saved-geo-model:/models/saved-geo-model/1 \
  -e MODEL_NAME="saved-geo-model" \
  tensorflow/serving:2.7.0
```