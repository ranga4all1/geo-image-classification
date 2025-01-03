# geo-image-classification
Classify geographical images

![Banner](images/Geo_Image_Classification_Banner.jpg)

## System Overview

Geographical image classification is crucial for various applications such as environmental monitoring, urban planning, and disaster management. This project aims to classify geographical images using advanced machine learning techniques.

The solution provides a **REST API interface** running via kubernetes that allows easy scaling and integration with existing systems, making it practical for real-world applications.

## Technical Architecture

The **Geo Image Classifier** uses deep learning techniques to classify geographical images. Key aspects include:

- **Multiple Model Evaluation**: The project evaluated several deep learning models including:
  - Convolutional Neural Networks (CNN)
  - Transfer Learning with pre-trained models
  
- **Model Selection**: After comprehensive testing and validation, **Xception** was selected as the final model due to its:
  - Superior accuracy
  - Ability to handle complex image features
  - Robustness against overfitting

- **Feature Engineering**: The model processes various types of input data:
  - Image data (satellite images, aerial photos)
  - Metadata (location, time of capture)
  - Other - regularization, data augmentation etc.

- **Practical Implementation**: The system provides:
  - Multiclass classification
  - REST API for easy integration with existing systems
  - Kubernetes implementation for easy scaling:
    - Separated image preprocessing: gateway deployment pod and service
    - Model inference: model deployment pod and service using TF-Serving
    - Gateway and model deployment and services can be scaled independently
    - Gateway can run on cpu nodes and model inference on gpu nodes 

## Data Specification

The dataset was sourced from:
1. [Kaggle](https://www.kaggle.com/datasets/puneet6060/intel-image-classification/data)

- Download dataset from [Kaggle](https://www.kaggle.com/datasets/puneet6060/intel-image-classification/data) into `data` folder
- Unzip data file

**Folder strutcure:**

- data
    - seg_train
    - seg_test
    - seg_pred

## Technical Stack

- **`Python 3.10`**
- **`conda`** for creating project level virtual environment 
- **`JupyterLab`** for experimentation
- **`TensorFlow`** for deep learning model development
- **`Flask`** for REST API interface
- **`Pipenv`** for virtual env (managing python dependencies)
- **`Docker`** for containerization (managing system dependencies)
- **`kind` and `kubectl`** for local kubernates deployment
- **`AWS EKS`** for cloud deployment


## Implementation Guide (Reproduce)

### Pre-requisites

1. A system with GPU is preferred for deep learning model training and experimentation
2. Docker, kind, kubectl

### Development Environment Configuration

1. Clone this repository:
```
git clone https://github.com/ranga4all1/geo-image-classification.git
cd geo-image-classification
```
2. Ceate conda environment
```
conda create -n geo python=3.10 numpy jupyter matplotlib
conda activate geo
pip install tensorflow==2.18 grpcio tensorflow-serving-api==2.18 keras-image-helper flask
```
3. Train deep learning model and save model file
```
cd code
python train.py
```
- Wait till model traing is finished and final model file named `geo-model.keras` is saved with weights for highest accuracy

4. Convert model to a special format called tensorflow `SavedModel` for use with `tf-serving`.
```
python saved-model.py
```
- This should save model in a directory structure similar to this:
```
── saved-geo-model
│   ├── assets
│   ├── fingerprint.pb
│   ├── saved_model.pb
│   └── variables
│       ├── variables.data-00000-of-00001
│       └── variables.index
```
5. Look at signature
```
saved_model_cli show --dir saved-geo-model --all | less
```
#### OR
```
saved_model_cli show --dir saved-geo-model --tag_set serve --signature_def serving_default
```

- signature should like similar to this:
```
The given SavedModel SignatureDef contains the following input(s):
  inputs['input_layer_1'] tensor_info:
      dtype: DT_FLOAT
      shape: (-1, 299, 299, 3)
      name: serving_default_input_layer_1:0
The given SavedModel SignatureDef contains the following output(s):
  outputs['output_0'] tensor_info:
      dtype: DT_FLOAT
      shape: (-1, 6)
      name: StatefulPartitionedCall_1:0
Method name is: tensorflow/serving/predict
```
- Note down values in square brackets: e. g. `inputs['input_layer_1']` and `outputs['output_0']`. You would need those to update a script next.

6. Update `gateway.py` file with your signature values and uncomment last few lines for testing e. g. It should look similar to this:
```
pb_request.inputs['input_layer_1'].CopyFrom(np_to_protobuf(X))
.
.
preds = pb_response.outputs['output_0'].float_val
.
.
if __name__ == '__main__':
    # url = 'https://github.com/ranga4all1/geo-image-classification/blob/main/images/glacier.jpg?raw=true'
    # response = predict(url)
    # print(response)
    app.run(debug=True, host='0.0.0.0', port=9696)
```
7. Update url used in 'test.py' file for use with docker local testing. Simply umcomment required url and comment the other. e.g.
```
url = "http://localhost:9696/predict"  # for use with docker local testing
```
8. Run the model (saved-geo-model) with the prebuilt docker image `tensorflow/serving:2.18.0`:

```
docker run -it --rm \
  -p 8500:8500 \
  -v $(pwd)/saved-geo-model:/models/saved-geo-model/1 \
  -e MODEL_NAME="saved-geo-model" \
  tensorflow/serving:2.18.0
```

#### Run gateway service flask application
```
python gateway.py
```
#### Test using 
```
python test.py
```
Result:
```
{'buildings': -0.7047825455665588, 'forest': -1.4779146909713745, 'glacier': 2.144595146179199, 'mountain': 1.1838254928588867, 'sea': -0.41370701789855957, 'street': -0.41950517892837524}
```
- Model classifies image as **Glacier**.


## Containerization using docker

1. Put everything in Pipenv
```
pip install pipenv
pipenv install grpcio keras-image-helper flask gunicorn tensorflow-protobuf==2.11.0
```
**Note:**
We will not install tensorflow in pipenv to keep our containers lightweight.

2. Build and Run model and gateway containers locally with docker-compose
```
docker-compose up -d
```
3. Test
```
python test.py
```
Result:
```
(geo-1) @ranga4all1 ➜ /workspaces/geo-image-classification/code (main) $ python test.py 
{'buildings': -0.7179785370826721, 'forest': -1.896227240562439, 'glacier': 2.369765043258667, 'mountain': 1.5169345140457153, 'sea': -0.49287453293800354, 'street': -0.46033942699432373}
```

## Kind: Local kubernetes

1. Create kind cluster
```
cd kube-config/

kind create cluster
```
#### Verify
``` 
kubectl cluster-info --context kind-kind
kubectl get service
docker ps
```
2. Load images to kind
```
kind load docker-image saved-geo-model:xception-001
kind load docker-image geo-gateway:001
```
3. Create gateway + model deployment and service
```
kubectl apply -f model-deployment.yaml
kubectl apply -f model-service.yaml
kubectl apply -f gateway-deployment.yaml
kubectl apply -f gateway-service.yaml
```
#### Verify
```
kubectl get pod
kubectl get service
```
Result:
```
(geo-1) @ranga4all1 ➜ /workspaces/geo-image-classification (main) $ kubectl get pod
NAME                                   READY   STATUS    RESTARTS        AGE
gateway-66fd949c59-d649f               1/1     Running   4 (8m17s ago)   23h
tf-serving-geo-model-b6584758b-mw6hh   1/1     Running   4 (8m17s ago)   24h

(geo-1) @ranga4all1 ➜ /workspaces/geo-image-classification (main) $ kubectl get service
NAME                   TYPE           CLUSTER-IP      EXTERNAL-IP   PORT(S)        AGE
gateway                LoadBalancer   10.96.194.104   <pending>     80:32610/TCP   23h
kubernetes             ClusterIP      10.96.0.1       <none>        443/TCP        24h
tf-serving-geo-model   ClusterIP      10.96.40.27     <none>        8500/TCP       23h
```

4. Test using: `kubectl port-forward service/gateway 8080:80` and replace the url on `test.py` to 8080 to get predictions.
```
kubectl port-forward service/gateway 8080:80
python test.py
```

Result:
```
(geo-1) @ranga4all1 ➜ /workspaces/geo-image-classification/code (main) $ python test.py 
{'buildings': -0.7179785370826721, 'forest': -1.896227240562439, 'glacier': 2.369765043258667, 'mountain': 1.5169345140457153, 'sea': -0.49287453293800354, 'street': -0.46033942699432373}
```


## Model Development and Analysis

For our experiments, we utilize Jupyter notebooks located in the [**`notebooks`**](notebooks/) folder.

### Available Notebooks:

- [**`notebook.ipynb`**](notebooks/notebook.ipynb)
  - **Data Preparation**
    - Exploratory Data Analysis (EDA)
      - Image count
      - Pre-process and view sample images

  - **Model Development**
    - Training the selected model using Xception
    - Parameter tuning, regularization, dropout, and data augmentation
    - Validation
  - **Model Save/Load**
    - Save the model in a `.keras` single file

- [**`use-model.ipynb`**](notebooks/use-model.ipynb)
  - Load and test the saved model


## Code

