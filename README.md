# 🚀 Kubernetes-Based Distributed LLM Inference API

## 📌 Overview
This project provides a **Kubernetes-based distributed inference pipeline** for **Large Language Models (LLMs)**. It deploys an **LLM inference API** as a FastAPI service inside a **Google Kubernetes Engine (GKE) Autopilot** cluster, ensuring scalability and high availability.

## 🎯 **Project Goals**
- **Deploy an LLM inference API** using Kubernetes on **GKE Autopilot**.
- **Enable autoscaling** using **Horizontal Pod Autoscaler (HPA)**.
- **Expose the API securely** with a **LoadBalancer service**.
- **Optimize model inference latency** for improved response times.

---

## 🏗️ **Tech Stack**
- **Kubernetes** (GKE Autopilot)
- **Docker** (Containerized API)
- **FastAPI** (Inference Server)
- **Hugging Face Transformers** (LLM Model)
- **Google Artifact Registry** (Container Image Storage)
- **Google Cloud Build** (Automated Image Builds)

---

## 🔥 **Benchmarking**

Running locally on Macbook M3 Pro, quantizing model with OpenVINO from HuggingFace achieved ~367 ms inference latency on GPT2 model. Cannot run PyTorch quantization locally, as it is not compatkble with Apple Silicon.



### Build Instructions

I am developing this on a Mac, but the code runs on a Linux container in Kubernetes. In order build the image correctly for this environment, I have to build it within a Linux environment. For this reason, I have been using CloudBuild from GCP to build my Docker image and then push it to Google's Artifact registry.

Steps:
1. login to GCP - `gcloud auth login`
2. `gcloud auth configure-docker`
3. set project id - `gcloud config set project <YOUR_PROJECT_ID>`
4. enable required services if not already enabled - `gcloud services enable cloudbuild.googleapis.com artifactregistry.googleapis.com`
5. Ensure docker is running locally.
6. Tag image - `docker tag <image_name> gcr.io/<YOUR_PROJECT_ID>/<image_name>:latest`
7. Build and push image to GCR using `cloudbuild.yaml` configuration file -> `gcloud builds submit --config Docker/cloudbuild.yaml`
- I already had this created from the first time I went throug this process. Please take a look at the format of my configuration file and adjust it to your needs.
- This step will take a ~10 minutes to complete
8. Restart deployment in GKE to pick up latest version of image - `kubectl rollout restart deployment <DEPLOYMENT_NAME>`

### Query API

The LLM Inference API is running in a GKE cluster with a Kubernetes service. To reach the API, you can curl the External IP of the Kubernetes service.

1. Get the External IP of Service - `kubectl get services`
```shell
(base) ivanmorrow@Ivans-MacBook-Pro k8s-distributed-llm-inference % kubectl get services
NAME                    TYPE           CLUSTER-IP       EXTERNAL-IP     PORT(S)        AGE
kubernetes              ClusterIP      34.118.224.1     <none>          443/TCP        18d
llm-inference-service   LoadBalancer   34.118.235.217   34.56.220.153   80:32745/TCP   11d
```
2. Use the External IP listed to run your curl command.

### Update Helm Chart

The helm chart resides in `llm-inference-chart/` directory. To make any changes to the helm chart, simply make your changes to any files in this directory. Then, run the following command to apply them: `helm upgrade llm-inference ./llm-inference-chart -f llm-inference-chart/values.yaml`

### Delete cluster

Can delete the cluster when you're not using it to save on GCP credits.

`gcloud container clusters delete <Cluster_name> --region <region>

To re-create the cluster, simply run: `gcloud container clusters create-auto <cluster_name> --region <region>`