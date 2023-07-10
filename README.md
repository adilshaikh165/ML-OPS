# ML-OPS Task for AI-PLANET Internship

## Project Title: Implementing Kubeflow with MLflow for Model Experiment Tracking

## Project Description:

Kubeflow is an open-source platform that makes it easy to deploy and
manage machine learning workflows on Kubernetes. MLflow is another open-source platform
that provides tools for tracking and managing machine learning experiments.
In this project, you will be setting up a Kubeflow cluster and integrating it with MLflow to track
the experiments run on the cluster. You will then use this setup to train and track the
performance of a model on a dataset.

## Project Tasks:

1. Install and configure Kubeflow on a Kubernetes/minikube cluster
2. Install and configure MLflow on the same cluster
3. Write a Python script to train a model on a dataset and log the experiment with MLflow
4. Use the Kubeflow pipeline system to run multiple experiments with different
hyperparameters and track them with MLflow
5. Compare the performance of the different models using the MLflow UI
6. Deploy the best model as a Kubernetes deployment and expose it as a service (this is
optional).

## Deliverables:

1. A Kubeflow cluster set up with MLflow integration
2. A Python script for training and logging model experiments with MLflow
3. A Kubeflow pipeline for running multiple experiments and tracking them with MLflow
4. A report summarizing the results of the experiments and the performance of the best
model
5. A Kubernetes deployment for the best model, along with instructions for accessing the
service (this is optional).

## Steps to be followed

### 1. Install and configure Kubeflow on Minikube cluster

### Prerequisites :

 - Docker — version 1.19
 - kubectl — version 1.15
 - minikube — version 1.15
   
### 1a) You can deploy the Kubeflow pipeline on Kubernetes/minikube cluster on Windows host machine powershell with administrative previliges using following few commands :

```bash
set PIPELINE_VERSION=2.0.0
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=$PIPELINE_VERSION"
kubectl wait --for condition=established --timeout=60s crd/applications.app.k8s.io
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/env/platform-agnostic-pns?ref=$PIPELINE_VERSION"
```

![Kubeflow Installation](https://github.com/adilshaikh165/ML-OPS/assets/98637502/f96c3981-222f-4994-866b-1fdd8075caba)

### 1b) Run the Kubectl command to view the Pods status

```bash
kubectl get pods -A
```
It'll show all the pods in the default as well as Kubeflow namespace.

![Kubeflow cmd config](https://github.com/adilshaikh165/ML-OPS/assets/98637502/bb4e95e1-42f1-49e6-9a05-5440d12dc626)

To view the Pods only from kubeflow namespace you can use following command :

```bash
kubectl get pods -n kubeflow
```

### 1c) Port-forward the kubeflow service to view kubeflow dashboard

Use the below command for port-forward :

```bash
 kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80
```

It'll give the local IP address through which we can view our kubeflow dashboard.

![Kubeflow port forward](https://github.com/adilshaikh165/ML-OPS/assets/98637502/de7c5c53-fcf6-438d-89bb-81104aecceef)


### 1d) Open the Web Browser and open localhost at port 8080

After opening the localhost:8080 you can view your Kubeflow dashboard.

![Kubeflow localhost](https://github.com/adilshaikh165/ML-OPS/assets/98637502/208a2826-1000-4d90-9c5c-0b41701c8f27)

Hence, we have successfully installed and configured the Kubeflow on our minikube cluster.

## 2. Install and configure the MLFLOW on the Minikube cluster

To integrate MLflow and kubeflow together on a Minikube cluster, follow these steps :

### 2a) Set up and start a Minikube cluster on your local machine.

### 2b) Install and Configure the Kubeflow on Minikube cluster which we have done already in Step 1.

### 2c) Install MLflow on your Minikube cluster. You can use Helm charts to simplify the installation process.

Run following few command's to install MLflow using Helm Charts :

```bash
helm repo add community-charts https://community-charts.github.io/helm-charts

helm install my-mlflow community-charts/mlflow --version 0.7.19
```

### 2d) Verify the installation and check the status of the MLflow deployment:

Use the following kubectl command to verify the installation :

```bash
kubectl get pods -n default
```

You'll get to see your mlflow pod up and running.

![MLFLOW verification](https://github.com/adilshaikh165/ML-OPS/assets/98637502/eae3a8a1-87c0-40c6-a441-34c9f0a20c53)

### 2e) Type "mlflow ui" in your Terminal

Once you type the "mlflow ui" in your terminal it'll give you the Localhost address for accessing your MLflow Dashboard.

![MLFLOW UI](https://github.com/adilshaikh165/ML-OPS/assets/98637502/8bf2e32f-250a-4cf8-87ca-d683cf92aa2f)

### 2f) Open the web browser and paste the Address got from the previous step

![MLFLOW Dashboard](https://github.com/adilshaikh165/ML-OPS/assets/98637502/2febe0ac-2b45-499f-8b5b-4c4a7e365c9c)

Hence, we have successfully Integrated Kubeflow and MLflow on our minikube cluster.

## 3. Setup Jupyter Notebooks

### 3a) Create Conda Environment to Open Jupyter notebook

 - Create conda environment

  ![Creating conda env](https://github.com/adilshaikh165/ML-OPS/assets/98637502/eb3e80a8-ec3d-4639-b5a7-2e10b37f9396)

  - Activate conda environment

    You can activate your conda environment created in the previous step using following command :
    
    ```bash
    conda activate <ENV_NAME>
    ```

    ![Activate conda env](https://github.com/adilshaikh165/ML-OPS/assets/98637502/0645f087-4169-4433-b8fa-3cf795b5fd4c)

   - Launch Jupyter Notebook from anaconda prompt

     Type the following command to launch jupyter notebook :

      ```bash
      jupyter notebook
      ```

      ![Jupyter notebook from cmd](https://github.com/adilshaikh165/ML-OPS/assets/98637502/b49ec5e1-838d-40a6-972b-93b5f675088a)

### 3b) Create the ".ipynb" file to write a Python script to train a model on a dataset and log the experiment with MLflow

At first, let's clone this repository so you have access to the code. You can use the terminal or directly do that in the browser.

```bash
git clone https://github.com/adilshaikh165/ML-OPS.git
```

Then open "MLOPS-INTERNSHIP-ASSESSMENT-TASK.ipynb" to get the gist of the Python Script which I have created to train a model on a "bank-full.csv" and log the experiment with MLflow.


## 4. Using MLflow UI to visualize and store all the Log's related to specific experiment

### 4a) Create experiment with basic classifier and records metrics

Refer the "create_experiment()" Function from the "MLOPS-INTERNSHIP-ASSESSMENT-TASK.ipynb" file.

It'll create and log a "basic classfier experiment" in the MLflow UI and will record all the relavant metrics as shown in the below few screenshots :

![Basic_classifier](https://github.com/adilshaikh165/ML-OPS/assets/98637502/3985b942-eff8-4964-9134-0af8c9fca7d1)

You can view all the relavant metrics, tags and artificats related to that perticular run.

![Basic_metrics_tags](https://github.com/adilshaikh165/ML-OPS/assets/98637502/8d8d3898-8dec-49d3-9cbc-80e229996478)

![CM](https://github.com/adilshaikh165/ML-OPS/assets/98637502/a7b3ef1f-5b41-4410-97ea-64c0211ec322)

![roc](https://github.com/adilshaikh165/ML-OPS/assets/98637502/e32212f0-50e0-430a-9a66-8b6541234041)


### 4b) Tune the ML Model using hyperparameters to increase it's accuracy

Refer the "hyper_parameter_tuning()" Function from the "MLOPS-INTERNSHIP-ASSESSMENT-TASK.ipynb" file.

It'll create and log a "Optimized Classifier Experiment" in the MLflow UI and will record all the relavant metrics as well as Parameters as shown in the below few screenshots :

![Optimized_ui](https://github.com/adilshaikh165/ML-OPS/assets/98637502/2828c24f-f9c4-4148-967a-55e3934b2369)

This time along with the Metrics, tags and artifcats you'll also get to log all hyper parameters, metrics, and artifacts which contains model, roc_auc curve PNG, confusion Matrix PNG Related to that Optimized Model.

![Optimized_params](https://github.com/adilshaikh165/ML-OPS/assets/98637502/4d727455-2a69-4b2d-8be0-78600b84a58d)

## 5. Creating the Ml Pipeline using Kubeflow Pipeline

Kubeflow Pipelines (KFP) is the most used component of Kubeflow. It allows you to create for every step or function in your ML project a reusable containerized pipeline component which can be chained together as a ML pipeline.

For the digits recognizer application, the pipeline is already created with the Python SDK. You can find the code in the file
```bash
kf-pipeline.ipynb
```

- Write a Python Function needed to train and predict
  
  We need to create a various functions in order to train and predict our ML Model. The various functions are prepare_data(), train_test_split() and
 training_basic_classifier(). You can find all these functions in "kf-pipeline.ipynb" file.

- Define the pipeline function and put together all the components

  ```bash
     @dsl.pipeline(
      name='Basic MLOPS classifier Kubeflow Demo Pipeline',
      description='A sample pipeline that performs IRIS classifier task'
     )
   
     def basic_classifier_pipeline(data_path: str):
       vop = dsl.VolumeOp(
       name="t-vol-1",
       resource_name="t-vol-1", 
       size="1Gi", 
       modes=dsl.VOLUME_MODE_RWO)
       
       prepare_data_task = create_step_prepare_data().add_pvolumes({data_path: vop.volume})
       train_test_split = create_step_train_test_split().add_pvolumes({data_path: vop.volume}).after(prepare_data_task)
       classifier_training = create_step_training_basic_classifier().add_pvolumes({data_path: vop.volume}).after(train_test_split)
       
       
       prepare_data_task.execution_options.caching_strategy.max_cache_staleness = "P0D"
       train_test_split.execution_options.caching_strategy.max_cache_staleness = "P0D"
       classifier_training.execution_options.caching_strategy.max_cache_staleness = "P0D"
       
  
    ```

- Mounting volume for component's output storage and binding this volume with all the components. The pipeline defines a volume named "t-vol-1" with a size of 1GiB. This volume is used to store the dataset and the model artifacts.

- Compiling pipeline and generating yaml

  Once the pipeline is complied the yaml file is automatically generated and it can be directly uploaded to kubeflow and create experiments and runs using UI. You can refer the sample yaml file in the GitHub repo named as "basic_classifier_pipeline_adil.yaml".

  ```bash
  kfp.compiler.Compiler().compile(
    pipeline_func=basic_classifier_pipeline,
    package_path='basic_classifier_pipeline_adil.yaml')
  ```

- Create a run from pipeline function using the code.
 
- Creation of the Persistent Volume
  
    ![Step1](https://github.com/adilshaikh165/ML-OPS/assets/98637502/74027b94-f1fc-4a02-80b6-18821f8273aa)

- Prepare Data for train-test split. prepare_data_task loads the dataset from a URL and saves it to a subdirectory called data in the pipeline's working directory.
  
    ![Step2](https://github.com/adilshaikh165/ML-OPS/assets/98637502/c6824c13-af6a-4300-bbb1-45cf2e0af2eb)

- Generation of train-test split. train_test_split splits the dataset into a training set and a test set.
  
    ![Train_test_split](https://github.com/adilshaikh165/ML-OPS/assets/98637502/b7d5aead-1d30-47fd-bde4-10e2920ae359)

- Training of Basic classifier model. classifier_training trains a logistic regression model on the training set. This step involves conversion of data type String into float for columns "job" and "married".

  I have mapped various attributes of the column "job" into the static float values and perform "One-hot" encoding on marital column.
  
    ![FinalPipeline](https://github.com/adilshaikh165/ML-OPS/assets/98637502/11325c33-fd6c-48c8-9069-b79226e1c8f5)



## 6. Kubernetes Deployment for the Best Model.(OPTIONAL)

Because of the time constraint I was not able to deploy the ML model on Kubernetes Cluster. But I was able to do it very easily because in my previous project I was done the same deployment only the difference was that was the Python Flask Application and not the ML Model. But the procedures are the same for deployment to the kubernetes cluster.
The best model in our case is "Optimized Model" from step "4b".

You can refer this blog of my previous project it contains all the deployment part to the kubernetes cluster and exposing it as service

The procedure will be some what like this:

![image](https://github.com/adilshaikh165/ML-OPS/assets/98637502/00b02108-2654-4803-83fb-f671efdb9544)


Blog Link : https://adilshaikh165.hashnode.dev/cloud-native-monitoring-application





