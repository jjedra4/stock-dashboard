# AWS ECS Fargate Deployment Guide

This guide outlines how to deploy the Stock Dashboard application using **AWS ECS (Elastic Container Service) on Fargate**. This architecture is serverless, scalable, and follows industry best practices for containerized applications.

## Architecture Overview

We will deploy two separate components using the same Docker image:

1.  **Dashboard Service**: A long-running web service hosting the Streamlit UI.
2.  **Scheduler Task**: A scheduled task (cron job) that runs daily to fetch data and retrain models.

### Why this architecture?
-   **Business Viable**: Uses standard container orchestration (ECS) rather than hobbyist platforms.
-   **Cost Effective**: Fargate charges only for the compute used. The Scheduler Task only incurs costs for the few minutes it runs each day.
-   **Scalable**: The Dashboard can be auto-scaled based on traffic.
-   **Maintainable**: Single Docker image for both workloads ensures consistency.

---

## Prerequisites

1.  **AWS CLI** installed and configured (`aws configure`).
2.  **Docker** installed and running.
3.  **Supabase** project URL and Key.
4.  **FMP API Key**.

---

## Step 1: Create ECR Repository

Create a registry to store your Docker images.

```bash
# 1. Create Repository
aws ecr create-repository --repository-name stock-dashboard --region us-east-1

# 2. Login Docker to AWS ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <YOUR_AWS_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com
```

## Step 2: Build and Push Image

```bash
# 1. Build
docker build --platform linux/amd64 -t stock-dashboard .

# 2. Tag
docker tag stock-dashboard:latest <YOUR_AWS_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/stock-dashboard:latest

# 3. Push
docker push <YOUR_AWS_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/stock-dashboard:latest
```
*Note: We use `--platform linux/amd64` to ensure compatibility with Fargate if building on an Apple Silicon Mac.*

---

## Step 3: Infrastructure Setup (ECS)

### 1. Create Cluster
Go to the AWS ECS Console and create a new **Cluster**.
-   **Infrastructure**: Fargate (Serverless)
-   **Name**: `stock-cluster`

### 2. Create Task Definition
Create a new **Task Definition** (JSON-based configuration for your container).
-   **Family**: `stock-task`
-   **Launch Type**: Fargate
-   **OS/Arch**: Linux / X86_64
-   **CPU**: 1 vCPU
-   **Memory**: 3 GB (Training needs RAM)
-   **Task Role**: `ecsTaskExecutionRole` (Ensure it has ECR pull permissions)
-   **Container**:
    -   **Name**: `app`
    -   **Image**: `<YOUR_ECR_URI>:latest`
    -   **Port Mappings**: 8501 (TCP)
    -   **Environment Variables**:
        -   `SUPABASE_URL`: value
        -   `SUPABASE_KEY`: value
        -   `FMP_API_KEY`: value
        -   `TICKERS`: `NVDA,AAPL,MSFT,GOOGL,AMZN,TSLA,META`

---

## Step 4: Deploy Dashboard Service

This keeps the UI running 24/7.

1.  In your Cluster, create a **Service**.
2.  **Compute Options**: Launch Type (Fargate).
3.  **Task Definition**: `stock-task` (Latest revision).
4.  **Service Name**: `dashboard-service`.
5.  **Desired Tasks**: 1.
6.  **Networking**:
    -   **VPC**: Select default VPC.
    -   **Subnets**: Select all public subnets.
    -   **Security Group**: Create new. Allow **Custom TCP** port **8501** from **Anywhere (0.0.0.0/0)**.
    -   **Auto-assign Public IP**: **ENABLED**.
7.  **Deploy**.

Once running, find the **Public IP** of the task in the "Tasks" tab. Access at `http://<PUBLIC_IP>:8501`.

---

## Step 5: Deploy Scheduler (Daily Job)

This runs the data ingestion pipeline once a day.

1.  Go to **Amazon EventBridge** > **Schedules**.
2.  **Create Schedule**.
3.  **Name**: `daily-stock-update`.
4.  **Schedule pattern**: Recurring schedule > Cron-based.
    -   Cron: `0 23 ? * MON-FRI *` (Runs at 11:00 PM UTC, Mon-Fri).
5.  **Target API**: **Amazon ECS** > **RunTask**.
6.  **Cluster**: `stock-cluster`.
7.  **Task Definition**: `stock-task`.
8.  **Launch Type**: Fargate.
9.  **Subnets**: Select public subnets.
10. **Security Groups**: Select the same group as dashboard (or one allowing outbound internet).
11. **Auto-assign Public IP**: ENABLED.
12. **Container Overrides**:
    -   Select your container (`app`).
    -   **Command Override**: `python`, `flows/daily_ingest.py`
    *Note: This command overrides the default `streamlit run...` command.*

---

## Summary

You now have a production-grade architecture:
-   **Dashboard**: Always available for users.
-   **Data Pipeline**: Runs automatically on a schedule, costing $0 when not running.
-   **Infrastructure**: Fully serverless and managed.
