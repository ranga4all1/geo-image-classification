# Geographical image classification

Classify geographical images using a Deep Learning image classifier REST API service built on kebernetes for easy scaling.

This project aims to classify geographical images into distinct categories using a deep learning-based image classification model. The solution is deployed as a REST API service, leveraging Kubernetes for scalability and robustness. The architecture supports high-throughput requests, ensuring rapid and accurate image classification even under varying loads. This scalable infrastructure facilitates integration with larger geospatial analytics systems, enabling seamless deployment in real-world environments.

Geographical image classification is crucial for various applications such as environmental monitoring, urban planning, and disaster management. 

![Banner](images/banner.jpg)

## Table of Contents

1. [System Overview](#system-overview)
2. [Technical Architecture](#technical-architecture)
3. [Data Specification](#data-specification)
4. [Technical Stack](#Technical-Stack)
5. [Implementation Guide (Reproduce)](#implementation-guide-reproduce)
6. [Cloud Deployment](#cloud-deployment)
7. [Model Development and Analysis](#model-development-and-analysis)
8. [Source Code](#source-code)
9. [Repository structure](#repository-structure)

## System Overview

This project aims to classify geographical images using advanced machine learning techniques.

The solution provides a **REST API interface** running via kubernetes that allows easy scaling and integration with existing systems, making it practical for real-world applications.

## Technical Architecture

The **Geo Image Classifier** uses deep learning techniques to classify geographical images. Key aspects include:

- **Multiple Variations Evaluation**: The project evaluated Transfer Learning with variations including:
  - With and without dropouts
  - With and without extra inner layers
  
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
- **`TensorFlow and Keras`** for deep learning model development
- **`Flask`** for REST API interface
- **`Pipenv`** for virtual env (managing python dependencies)
- **`Docker`** for containerization (managing system dependencies)
- **`kind` and `kubectl`** for local kubernates deployment
- **`EKS`** for deploying to Elastic Kubernetes Service (EKS) cluster on AWS Cloud


## Implementation Guide (Reproduce)

### Pre-requisites

1. A system with GPU is preferred for deep learning model training and experimentation
2. Conda, Docker, kind, kubectl, eksctl


### Development Environment Configuration

1. Clone this repository:
```
git clone https://github.com/ranga4all1/geo-image-classification.git
cd geo-image-classification
```
  - Download dataset from [Kaggle](https://www.kaggle.com/datasets/puneet6060/intel-image-classification/data) into `data` folder
  - Unzip data file

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

## Cloud deployment

1. Create eks cluster: 
```
eksctl create cluster -f eks-config.yaml
```
2. Publish local docker images to ECR:

- Create aws ecr repository for eks cluster: `aws ecr create-repository --repository-name geo-model-images`
- Bash commands to run in the teminal to push docker images to ecr repository:
```
# Registry URI
ACCOUNT_ID=<your-account-id>
REGION=us-west-2
REGISTRY_NAME=geo-model-images
PREFIX=${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${REGISTRY_NAME}

# Tag local docker images to remote tag
GATEWAY_LOCAL=geo-gateway:001 # Gateway
GATEWAY_REMOTE=${PREFIX}:geo-gateway-001 # notice the ':' is replaced with '-'
docker tag ${GATEWAY_LOCAL} ${GATEWAY_REMOTE}

MODEL_LOCAL=saved-geo-model:xception-001 # tf-serving model
MODEL_REMOTE=${PREFIX}:saved-geo-model-xception-001 # same thing ':' is replaced with '-' before xception
docker tag ${MODEL_LOCAL} ${MODEL_REMOTE}

# Push tagged docker images
docker push ${MODEL_REMOTE}
docker push ${GATEWAY_REMOTE}
```

- Login to ecr and push images: `$(aws ecr get-login --no-include-email)`, first push the model and then gateway remote image.
- Get the uri of these images `echo ${MODEL_REMOTE}` and `echo ${GATEWAY_REMOTE}` and add them to `model-deployment.yaml` and `gateway-deployment.yaml` respectively.

3. Apply all the yaml config files to remote node coming from eks (`kubectl get nodes`):
```
kubectl apply -f model-deployment.yaml
kubectl apply -f model-service.yaml
kubectl apply -f gateway-deployment.yaml
kubectl apply -f gateway-service.yaml
```
- Testing the deployment pods and services should give us predictions.

4. Executing `kubectl get service` should give us the external port address which need to add in the `test.py` as access url for predictions (e.g., url = 'http://a3399e***-5180***.us-west-2-123.elb.amazonaws.com/predict').

- In AWS Console GUI, you should be able to see ECR images, EC2 instances created and load balancer/DNS name for service.

5. To delete the remote cluster: `eksctl delete cluster --name geo-model-eks`


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


## Source Code

The main application scripts are located in the [**`code`**](code/) folder. Below are the key files:

- [**`gateway.py`**](code/gateway.py) - Gateway service for handling requests
- [**`proto.py`**](code/proto.py) - Protocol buffer definitions
- [**`saved-model.py`**](code/saved-model.py) - Script to convert Keras model to TensorFlow SavedModel format
- [**`test-geo-model.py`**](code/test-geo-model.py) - An optional test script to test/use the geographical model in `.keras` format. This can be used as stepping stone if you decide to deploy `.keras` model in severless - for inferencing single image or as batch processing.
- [**`test.py`**](code/test.py) - General test script
- [**`train.py`**](code/train.py) - Script to train the model
- [**`docker-compose.yaml`**](code/docker-compose.yaml) - Docker Compose configuration for running services
- [**`image-gateway.dockerfile`**](code/image-gateway.dockerfile) - Dockerfile for building the gateway image
- [**`image-model.dockerfile`**](code/image-model.dockerfile) - Dockerfile for building the model image
- [**`Pipfile`**](code/Pipfile) - Pipenv configuration file for managing dependencies
- [**`Pipfile.lock`**](code/Pipfile.lock) - Pipenv lock file for dependency management

Additional files and directories:
- [**`geo-model.keras`**](code/geo-model.keras) - Pre-trained Keras model file
- [**`kube-config/`**](code/kube-config/) - Kubernetes configuration files
  - [**`gateway-deployment`**](code/kube-config/gateway-deployment.yaml) - Kubernetes deployment config for gateway
  - [**`gateway-service`**](code/kube-config/gateway-service.yaml) - Kubernetes service config for gateway
  - [**`model-deployment`**](code/kube-config/model-deployment.yaml) - Kubernetes deployment config for tf-serving model
  - [**`model-service`**](code/kube-config/model-service.yaml) - Kubernetes service config for tf-serving model
  - [**`eks-config.yaml`**](code/kube-config/eks-config.yaml) - EKS cluster config

  
## Repository structure
```
.
├── LICENSE
├── README.md
├── code
│   ├── Pipfile
│   ├── Pipfile.lock
│   ├── docker-compose.yaml
│   ├── gateway.py
│   ├── geo-model.keras
│   ├── image-gateway.dockerfile
│   ├── image-model.dockerfile
│   ├── kube-config
│   │   ├── eks-config.yaml
│   │   ├── gateway-deployment.yaml
│   │   ├── gateway-service.yaml
│   │   ├── model-deployment.yaml
│   │   └── model-service.yaml
│   ├── proto.py
│   ├── saved-model.py
│   ├── test-geo-model.py
│   ├── test.ipynb
│   ├── test.py
│   ├── tf-serving-connect.ipynb
│   └── train.py
├── commands.md
├── data
│   └── DATA.md
├── images
│   ├── banner.jpg
│   ├── building-1.jpg
│   ├── building.jpg
│   └── glacier.jpg
└── notebooks
    ├── notebook.ipynb
    ├── use-model.ipynb
    ├── xception_v1_09_0.904.keras
    └── xception_v4_1_11_0.927.keras
```
