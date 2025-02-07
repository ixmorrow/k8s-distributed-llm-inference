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
