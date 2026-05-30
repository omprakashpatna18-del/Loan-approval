{
  "metadata": {
    "kernelspec": {
      "language": "python",
      "display_name": "Python 3",
      "name": "python3"
    },
    "language_info": {
      "name": "python",
      "version": "3.12.13",
      "mimetype": "text/x-python",
      "codemirror_mode": {
        "name": "ipython",
        "version": 3
      },
      "pygments_lexer": "ipython3",
      "nbconvert_exporter": "python",
      "file_extension": ".py"
    },
    "kaggle": {
      "accelerator": "none",
      "dataSources": [
        {
          "sourceType": "datasetVersion",
          "sourceId": 6144787,
          "datasetId": 3523739,
          "databundleVersionId": 6223645
        }
      ],
      "dockerImageVersionId": 31400,
      "isInternetEnabled": true,
      "language": "python",
      "sourceType": "notebook",
      "isGpuEnabled": false
    },
    "colab": {
      "name": "Loan_approval",
      "provenance": [],
      "include_colab_link": true
    }
  },
  "nbformat_minor": 0,
  "nbformat": 4,
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/omprakashpatna18-del/Loan-approval/blob/main/Loan_approval.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "source": [
        "# IMPORTANT: RUN THIS CELL IN ORDER TO IMPORT YOUR KAGGLE DATA SOURCES,\n",
        "# THEN FEEL FREE TO DELETE THIS CELL.\n",
        "# NOTE: THIS NOTEBOOK ENVIRONMENT DIFFERS FROM KAGGLE'S PYTHON\n",
        "# ENVIRONMENT SO THERE MAY BE MISSING LIBRARIES USED BY YOUR\n",
        "# NOTEBOOK.\n",
        "import kagglehub\n",
        "architsharma01_loan_approval_prediction_dataset_path = kagglehub.dataset_download('architsharma01/loan-approval-prediction-dataset')\n",
        "\n",
        "print('Data source import complete.')\n"
      ],
      "metadata": {
        "id": "t7oN3WsaKKAj"
      },
      "cell_type": "code",
      "outputs": [],
      "execution_count": null
    },
    {
      "cell_type": "code",
      "source": [
        "import pandas as pd\n",
        "import numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n"
      ],
      "metadata": {
        "trusted": true,
        "execution": {
          "iopub.status.busy": "2026-05-30T13:04:42.912809Z",
          "iopub.execute_input": "2026-05-30T13:04:42.913588Z",
          "iopub.status.idle": "2026-05-30T13:04:42.919171Z",
          "shell.execute_reply.started": "2026-05-30T13:04:42.913555Z",
          "shell.execute_reply": "2026-05-30T13:04:42.918314Z"
        },
        "id": "hh6hf6xeKKAp"
      },
      "outputs": [],
      "execution_count": null
    },
    {
      "cell_type": "code",
      "source": [
        "import warnings\n",
        "warnings.filterwarnings(\"ignore\")"
      ],
      "metadata": {
        "trusted": true,
        "execution": {
          "iopub.status.busy": "2026-05-30T13:04:45.775101Z",
          "iopub.execute_input": "2026-05-30T13:04:45.775868Z",
          "iopub.status.idle": "2026-05-30T13:04:45.780026Z",
          "shell.execute_reply.started": "2026-05-30T13:04:45.775818Z",
          "shell.execute_reply": "2026-05-30T13:04:45.778907Z"
        },
        "id": "YSSst-NJKKAr"
      },
      "outputs": [],
      "execution_count": null
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.model_selection import(train_test_split,cross_val_score,RandomizedSearchCV)\n",
        "from sklearn.pipeline import Pipeline\n",
        "from sklearn.compose import ColumnTransformer\n",
        "from sklearn.impute import SimpleImputer\n",
        "from sklearn.preprocessing import (StandardScaler, OneHotEncoder)\n",
        "from sklearn.linear_model import LogisticRegression\n",
        "from sklearn.tree import DecisionTreeClassifier\n",
        "from sklearn.ensemble import (\n",
        "RandomForestClassifier,\n",
        "GradientBoostingClassifier)\n",
        "\n",
        "# Metrics\n",
        "from sklearn.metrics import (\n",
        "accuracy_score,\n",
        "auc,\n",
        "classification_report,\n",
        "confusion_matrix,\n",
        "roc_curve,\n",
        "roc_auc_score)"
      ],
      "metadata": {
        "trusted": true,
        "execution": {
          "iopub.status.busy": "2026-05-30T13:04:50.003486Z",
          "iopub.execute_input": "2026-05-30T13:04:50.004262Z",
          "iopub.status.idle": "2026-05-30T13:04:50.865307Z",
          "shell.execute_reply.started": "2026-05-30T13:04:50.004231Z",
          "shell.execute_reply": "2026-05-30T13:04:50.864434Z"
        },
        "id": "UcENckQ2KKAs"
      },
      "outputs": [],
      "execution_count": null
    },
    {
      "cell_type": "code",
      "source": [
        "data=pd.read_csv(\"/kaggle/input/datasets/architsharma01/loan-approval-prediction-dataset/loan_approval_dataset.csv\")\n",
        "print(data)"
      ],
      "metadata": {
        "trusted": true,
        "execution": {
          "iopub.status.busy": "2026-05-30T13:04:54.91377Z",
          "iopub.execute_input": "2026-05-30T13:04:54.914321Z",
          "iopub.status.idle": "2026-05-30T13:04:54.967215Z",
          "shell.execute_reply.started": "2026-05-30T13:04:54.914292Z",
          "shell.execute_reply": "2026-05-30T13:04:54.965936Z"
        },
        "id": "qUNJ3QqZKKAt"
      },
      "outputs": [],
      "execution_count": null
    },
    {
      "cell_type": "code",
      "source": [
        "data.describe()"
      ],
      "metadata": {
        "trusted": true,
        "execution": {
          "iopub.status.busy": "2026-05-30T13:05:00.184232Z",
          "iopub.execute_input": "2026-05-30T13:05:00.184992Z",
          "iopub.status.idle": "2026-05-30T13:05:00.232466Z",
          "shell.execute_reply.started": "2026-05-30T13:05:00.184962Z",
          "shell.execute_reply": "2026-05-30T13:05:00.231544Z"
        },
        "id": "hpnw4hXeKKAu"
      },
      "outputs": [],
      "execution_count": null
    },
    {
      "cell_type": "code",
      "source": [
        "data.isnull().sum()"
      ],
      "metadata": {
        "trusted": true,
        "execution": {
          "iopub.status.busy": "2026-05-30T13:05:02.913719Z",
          "iopub.execute_input": "2026-05-30T13:05:02.914204Z",
          "iopub.status.idle": "2026-05-30T13:05:02.924803Z",
          "shell.execute_reply.started": "2026-05-30T13:05:02.914174Z",
          "shell.execute_reply": "2026-05-30T13:05:02.923607Z"
        },
        "id": "Krde-8joKKAv"
      },
      "outputs": [],
      "execution_count": null
    },
    {
      "cell_type": "code",
      "source": [
        "print(data.duplicated().sum())"
      ],
      "metadata": {
        "trusted": true,
        "execution": {
          "iopub.status.busy": "2026-05-30T09:45:28.158941Z",
          "iopub.execute_input": "2026-05-30T09:45:28.159324Z",
          "iopub.status.idle": "2026-05-30T09:45:28.172356Z",
          "shell.execute_reply.started": "2026-05-30T09:45:28.159297Z",
          "shell.execute_reply": "2026-05-30T09:45:28.171517Z"
        },
        "id": "Y-RyfxX0KKAw"
      },
      "outputs": [],
      "execution_count": null
    },
    {
      "cell_type": "code",
      "source": [
        "data.columns=(data.columns.str.strip().str.replace(\" \",\"_\"))\n",
        "data.columns"
      ],
      "metadata": {
        "trusted": true,
        "execution": {
          "iopub.status.busy": "2026-05-30T13:05:06.003449Z",
          "iopub.execute_input": "2026-05-30T13:05:06.003756Z",
          "iopub.status.idle": "2026-05-30T13:05:06.011233Z",
          "shell.execute_reply.started": "2026-05-30T13:05:06.003732Z",
          "shell.execute_reply": "2026-05-30T13:05:06.01017Z"
        },
        "id": "9EPCDXMWKKAx"
      },
      "outputs": [],
      "execution_count": null
    },
    {
      "cell_type": "code",
      "source": [
        "plt.figure(figsize=(10,5))\n",
        "sns.countplot(x=\"loan_status\",data=data)\n"
      ],
      "metadata": {
        "trusted": true,
        "execution": {
          "iopub.status.busy": "2026-05-30T13:05:09.273564Z",
          "iopub.execute_input": "2026-05-30T13:05:09.273882Z",
          "iopub.status.idle": "2026-05-30T13:05:09.502603Z",
          "shell.execute_reply.started": "2026-05-30T13:05:09.273858Z",
          "shell.execute_reply": "2026-05-30T13:05:09.501908Z"
        },
        "id": "l8j24Q8yKKAy"
      },
      "outputs": [],
      "execution_count": null
    },
    {
      "cell_type": "code",
      "source": [
        "plt.figure(figsize=(10,5))\n",
        "sns.histplot(data[\"income_annum\"],bins=30,kde=True,color=\"red\")\n",
        "plt.show"
      ],
      "metadata": {
        "trusted": true,
        "execution": {
          "iopub.status.busy": "2026-05-30T13:05:12.018674Z",
          "iopub.execute_input": "2026-05-30T13:05:12.01957Z",
          "iopub.status.idle": "2026-05-30T13:05:12.286787Z",
          "shell.execute_reply.started": "2026-05-30T13:05:12.019538Z",
          "shell.execute_reply": "2026-05-30T13:05:12.285894Z"
        },
        "id": "UMWIfcQnKKAy"
      },
      "outputs": [],
      "execution_count": null
    },
    {
      "cell_type": "code",
      "source": [
        "plt.figure(figsize=(10,5))\n",
        "sns.boxplot(x=\"loan_status\",y=\"loan_amount\",data=data,color=\"cyan\")"
      ],
      "metadata": {
        "trusted": true,
        "execution": {
          "iopub.status.busy": "2026-05-30T13:05:17.113686Z",
          "iopub.execute_input": "2026-05-30T13:05:17.114131Z",
          "iopub.status.idle": "2026-05-30T13:05:17.278508Z",
          "shell.execute_reply.started": "2026-05-30T13:05:17.114101Z",
          "shell.execute_reply": "2026-05-30T13:05:17.277335Z"
        },
        "id": "4WRWCnbWKKAz"
      },
      "outputs": [],
      "execution_count": null
    },
    {
      "cell_type": "code",
      "source": [
        "plt.figure(figsize=(8,5))\n",
        "sns.countplot(x=\"education\",hue=\"loan_status\",data=data)"
      ],
      "metadata": {
        "trusted": true,
        "execution": {
          "iopub.status.busy": "2026-05-30T09:45:43.639485Z",
          "iopub.execute_input": "2026-05-30T09:45:43.640251Z",
          "iopub.status.idle": "2026-05-30T09:45:43.84439Z",
          "shell.execute_reply.started": "2026-05-30T09:45:43.640217Z",
          "shell.execute_reply": "2026-05-30T09:45:43.843462Z"
        },
        "id": "7wgvZ2GTKKA0"
      },
      "outputs": [],
      "execution_count": null
    },
    {
      "cell_type": "code",
      "source": [
        "plt.figure(figsize=(10,5))\n",
        "corr=data.corr(numeric_only=True)\n",
        "sns.heatmap(corr,annot=True,cmap=\"coolwarm\")"
      ],
      "metadata": {
        "trusted": true,
        "execution": {
          "iopub.status.busy": "2026-05-30T13:05:23.234913Z",
          "iopub.execute_input": "2026-05-30T13:05:23.23525Z",
          "iopub.status.idle": "2026-05-30T13:05:23.727239Z",
          "shell.execute_reply.started": "2026-05-30T13:05:23.235223Z",
          "shell.execute_reply": "2026-05-30T13:05:23.725915Z"
        },
        "id": "vOkA0RO4KKA0"
      },
      "outputs": [],
      "execution_count": null
    },
    {
      "cell_type": "code",
      "source": [
        "data=data.drop(\"loan_id\",axis=1)"
      ],
      "metadata": {
        "trusted": true,
        "execution": {
          "iopub.status.busy": "2026-05-30T13:05:30.683784Z",
          "iopub.execute_input": "2026-05-30T13:05:30.684129Z",
          "iopub.status.idle": "2026-05-30T13:05:30.694344Z",
          "shell.execute_reply.started": "2026-05-30T13:05:30.684103Z",
          "shell.execute_reply": "2026-05-30T13:05:30.693319Z"
        },
        "id": "PtJhIdYZKKA1"
      },
      "outputs": [],
      "execution_count": null
    },
    {
      "cell_type": "code",
      "source": [
        "data"
      ],
      "metadata": {
        "trusted": true,
        "execution": {
          "iopub.status.busy": "2026-05-30T13:05:33.143897Z",
          "iopub.execute_input": "2026-05-30T13:05:33.14427Z",
          "iopub.status.idle": "2026-05-30T13:05:33.157435Z",
          "shell.execute_reply.started": "2026-05-30T13:05:33.144241Z",
          "shell.execute_reply": "2026-05-30T13:05:33.156286Z"
        },
        "id": "PvniG-QmKKA1"
      },
      "outputs": [],
      "execution_count": null
    },
    {
      "cell_type": "code",
      "source": [
        "\n",
        "data[\"loan_status\"]=data[\"loan_status\"].str.strip()\n",
        "y=data[\"loan_status\"]"
      ],
      "metadata": {
        "trusted": true,
        "execution": {
          "iopub.status.busy": "2026-05-30T13:05:37.043544Z",
          "iopub.execute_input": "2026-05-30T13:05:37.043886Z",
          "iopub.status.idle": "2026-05-30T13:05:37.050751Z",
          "shell.execute_reply.started": "2026-05-30T13:05:37.04384Z",
          "shell.execute_reply": "2026-05-30T13:05:37.04958Z"
        },
        "id": "PxDqdCf9KKA1"
      },
      "outputs": [],
      "execution_count": null
    },
    {
      "cell_type": "code",
      "source": [
        "X=data.drop(\"loan_status\",axis=1)\n",
        "data[\"loan_status\"]=data[\"loan_status\"].str.strip()\n",
        "y=data[\"loan_status\"].map({\"Approved\":1,\"Rejected\":0})"
      ],
      "metadata": {
        "trusted": true,
        "execution": {
          "iopub.status.busy": "2026-05-30T13:05:40.819705Z",
          "iopub.execute_input": "2026-05-30T13:05:40.820165Z",
          "iopub.status.idle": "2026-05-30T13:05:40.829774Z",
          "shell.execute_reply.started": "2026-05-30T13:05:40.820137Z",
          "shell.execute_reply": "2026-05-30T13:05:40.82855Z"
        },
        "id": "egw4Ea7SKKA2"
      },
      "outputs": [],
      "execution_count": null
    },
    {
      "cell_type": "code",
      "source": [
        "categorical=X.select_dtypes(include=[\"object\"]).columns.tolist()\n",
        "numerical=X.select_dtypes(include=[\"int64\",\"float64\"]).columns.tolist()"
      ],
      "metadata": {
        "trusted": true,
        "execution": {
          "iopub.status.busy": "2026-05-30T13:05:43.710237Z",
          "iopub.execute_input": "2026-05-30T13:05:43.711156Z",
          "iopub.status.idle": "2026-05-30T13:05:43.717104Z",
          "shell.execute_reply.started": "2026-05-30T13:05:43.711125Z",
          "shell.execute_reply": "2026-05-30T13:05:43.71583Z"
        },
        "id": "AWS1_PzJKKA2"
      },
      "outputs": [],
      "execution_count": null
    },
    {
      "cell_type": "code",
      "source": [
        "print(categorical)\n",
        "print(numerical)"
      ],
      "metadata": {
        "trusted": true,
        "execution": {
          "iopub.status.busy": "2026-05-30T13:05:46.173776Z",
          "iopub.execute_input": "2026-05-30T13:05:46.174168Z",
          "iopub.status.idle": "2026-05-30T13:05:46.17895Z",
          "shell.execute_reply.started": "2026-05-30T13:05:46.174139Z",
          "shell.execute_reply": "2026-05-30T13:05:46.177915Z"
        },
        "id": "i2pBvuSOKKA3"
      },
      "outputs": [],
      "execution_count": null
    },
    {
      "cell_type": "code",
      "source": [
        "#numerical pipeline\n",
        "numeric_pipeline=Pipeline(steps=[(\"imputer\",SimpleImputer(strategy=\"median\")),(\"scaler\",StandardScaler())])"
      ],
      "metadata": {
        "trusted": true,
        "execution": {
          "iopub.status.busy": "2026-05-30T13:05:50.213699Z",
          "iopub.execute_input": "2026-05-30T13:05:50.214651Z",
          "iopub.status.idle": "2026-05-30T13:05:50.221785Z",
          "shell.execute_reply.started": "2026-05-30T13:05:50.214618Z",
          "shell.execute_reply": "2026-05-30T13:05:50.220819Z"
        },
        "id": "GnNgYikbKKA3"
      },
      "outputs": [],
      "execution_count": null
    },
    {
      "cell_type": "code",
      "source": [
        "numeric_pipeline"
      ],
      "metadata": {
        "trusted": true,
        "execution": {
          "iopub.status.busy": "2026-05-30T13:05:53.403828Z",
          "iopub.execute_input": "2026-05-30T13:05:53.404595Z",
          "iopub.status.idle": "2026-05-30T13:05:53.421997Z",
          "shell.execute_reply.started": "2026-05-30T13:05:53.404565Z",
          "shell.execute_reply": "2026-05-30T13:05:53.421033Z"
        },
        "id": "1r7Ziq9GKKA4"
      },
      "outputs": [],
      "execution_count": null
    },
    {
      "cell_type": "code",
      "source": [
        "category_pipeline=Pipeline(steps=[(\"imputer\",SimpleImputer(strategy=\"most_frequent\")),(\"encoder\",OneHotEncoder(handle_unknown=\"ignore\"))])"
      ],
      "metadata": {
        "trusted": true,
        "execution": {
          "iopub.status.busy": "2026-05-30T13:05:57.453536Z",
          "iopub.execute_input": "2026-05-30T13:05:57.454382Z",
          "iopub.status.idle": "2026-05-30T13:05:57.459108Z",
          "shell.execute_reply.started": "2026-05-30T13:05:57.45435Z",
          "shell.execute_reply": "2026-05-30T13:05:57.458124Z"
        },
        "id": "kjMLDPZTKKA4"
      },
      "outputs": [],
      "execution_count": null
    },
    {
      "cell_type": "code",
      "source": [
        "category_pipeline"
      ],
      "metadata": {
        "trusted": true,
        "execution": {
          "iopub.status.busy": "2026-05-30T13:06:00.973873Z",
          "iopub.execute_input": "2026-05-30T13:06:00.974178Z",
          "iopub.status.idle": "2026-05-30T13:06:00.985952Z",
          "shell.execute_reply.started": "2026-05-30T13:06:00.974153Z",
          "shell.execute_reply": "2026-05-30T13:06:00.984177Z"
        },
        "id": "BHvbYpehKKA5"
      },
      "outputs": [],
      "execution_count": null
    },
    {
      "cell_type": "code",
      "source": [
        "preprocessor=ColumnTransformer(\n",
        "    transformers=[(\"num\",numeric_pipeline,numerical),\n",
        "                (\"cat\", category_pipeline,categorical)]\n",
        ")"
      ],
      "metadata": {
        "trusted": true,
        "execution": {
          "iopub.status.busy": "2026-05-30T13:06:05.873575Z",
          "iopub.execute_input": "2026-05-30T13:06:05.873888Z",
          "iopub.status.idle": "2026-05-30T13:06:05.878733Z",
          "shell.execute_reply.started": "2026-05-30T13:06:05.873863Z",
          "shell.execute_reply": "2026-05-30T13:06:05.877694Z"
        },
        "id": "oja-MLppKKA5"
      },
      "outputs": [],
      "execution_count": null
    },
    {
      "cell_type": "code",
      "source": [
        "preprocessor"
      ],
      "metadata": {
        "trusted": true,
        "execution": {
          "iopub.status.busy": "2026-05-30T13:06:09.883496Z",
          "iopub.execute_input": "2026-05-30T13:06:09.883804Z",
          "iopub.status.idle": "2026-05-30T13:06:09.910192Z",
          "shell.execute_reply.started": "2026-05-30T13:06:09.883778Z",
          "shell.execute_reply": "2026-05-30T13:06:09.909129Z"
        },
        "id": "wGZxgdACKKA6"
      },
      "outputs": [],
      "execution_count": null
    },
    {
      "cell_type": "code",
      "source": [
        "X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)"
      ],
      "metadata": {
        "trusted": true,
        "execution": {
          "iopub.status.busy": "2026-05-30T13:06:13.209464Z",
          "iopub.execute_input": "2026-05-30T13:06:13.209814Z",
          "iopub.status.idle": "2026-05-30T13:06:13.222863Z",
          "shell.execute_reply.started": "2026-05-30T13:06:13.209786Z",
          "shell.execute_reply": "2026-05-30T13:06:13.221985Z"
        },
        "id": "rHO9hrPwKKA6"
      },
      "outputs": [],
      "execution_count": null
    },
    {
      "cell_type": "code",
      "source": [
        "model=GradientBoostingClassifier(random_state=42)"
      ],
      "metadata": {
        "trusted": true,
        "execution": {
          "iopub.status.busy": "2026-05-30T13:06:18.583684Z",
          "iopub.execute_input": "2026-05-30T13:06:18.584052Z",
          "iopub.status.idle": "2026-05-30T13:06:18.588464Z",
          "shell.execute_reply.started": "2026-05-30T13:06:18.584024Z",
          "shell.execute_reply": "2026-05-30T13:06:18.587504Z"
        },
        "id": "pQ29X44mKKA6"
      },
      "outputs": [],
      "execution_count": null
    },
    {
      "cell_type": "code",
      "source": [
        "pipeline=Pipeline(steps=[(\"preprocessor\",preprocessor),(\"model\",model)])"
      ],
      "metadata": {
        "trusted": true,
        "execution": {
          "iopub.status.busy": "2026-05-30T13:06:22.423634Z",
          "iopub.execute_input": "2026-05-30T13:06:22.424333Z",
          "iopub.status.idle": "2026-05-30T13:06:22.428829Z",
          "shell.execute_reply.started": "2026-05-30T13:06:22.424301Z",
          "shell.execute_reply": "2026-05-30T13:06:22.427681Z"
        },
        "id": "Nqb-e1_UKKA7"
      },
      "outputs": [],
      "execution_count": null
    },
    {
      "cell_type": "code",
      "source": [
        "pipeline.fit(X_train,y_train)"
      ],
      "metadata": {
        "trusted": true,
        "execution": {
          "iopub.status.busy": "2026-05-30T13:06:34.904333Z",
          "iopub.execute_input": "2026-05-30T13:06:34.905242Z",
          "iopub.status.idle": "2026-05-30T13:06:35.777553Z",
          "shell.execute_reply.started": "2026-05-30T13:06:34.90521Z",
          "shell.execute_reply": "2026-05-30T13:06:35.776589Z"
        },
        "id": "yABFAzG1KKA7"
      },
      "outputs": [],
      "execution_count": null
    },
    {
      "cell_type": "code",
      "source": [
        "y_pred=pipeline.predict(X_test)"
      ],
      "metadata": {
        "trusted": true,
        "execution": {
          "iopub.status.busy": "2026-05-30T13:06:39.933341Z",
          "iopub.execute_input": "2026-05-30T13:06:39.93364Z",
          "iopub.status.idle": "2026-05-30T13:06:39.950732Z",
          "shell.execute_reply.started": "2026-05-30T13:06:39.933616Z",
          "shell.execute_reply": "2026-05-30T13:06:39.950033Z"
        },
        "id": "Ap3BRaFHKKA7"
      },
      "outputs": [],
      "execution_count": null
    },
    {
      "cell_type": "code",
      "source": [
        "accuracy = accuracy_score(y_test, y_pred)"
      ],
      "metadata": {
        "trusted": true,
        "execution": {
          "iopub.status.busy": "2026-05-30T13:06:51.229783Z",
          "iopub.execute_input": "2026-05-30T13:06:51.230679Z",
          "iopub.status.idle": "2026-05-30T13:06:51.236782Z",
          "shell.execute_reply.started": "2026-05-30T13:06:51.230649Z",
          "shell.execute_reply": "2026-05-30T13:06:51.235789Z"
        },
        "id": "adAntA1dKKA8"
      },
      "outputs": [],
      "execution_count": null
    },
    {
      "cell_type": "code",
      "source": [
        "roc_auc = roc_auc_score(y_test,y_pred)"
      ],
      "metadata": {
        "trusted": true,
        "execution": {
          "iopub.status.busy": "2026-05-30T13:06:54.431302Z",
          "iopub.execute_input": "2026-05-30T13:06:54.431901Z",
          "iopub.status.idle": "2026-05-30T13:06:54.440222Z",
          "shell.execute_reply.started": "2026-05-30T13:06:54.431868Z",
          "shell.execute_reply": "2026-05-30T13:06:54.439473Z"
        },
        "id": "N9OyHnXVKKA8"
      },
      "outputs": [],
      "execution_count": null
    },
    {
      "cell_type": "code",
      "source": [
        "cv_score = cross_val_score(pipeline,X_train,y_train,cv=5,scoring='accuracy').mean()"
      ],
      "metadata": {
        "trusted": true,
        "execution": {
          "iopub.status.busy": "2026-05-30T13:06:57.944174Z",
          "iopub.execute_input": "2026-05-30T13:06:57.944527Z",
          "iopub.status.idle": "2026-05-30T13:07:01.440067Z",
          "shell.execute_reply.started": "2026-05-30T13:06:57.9445Z",
          "shell.execute_reply": "2026-05-30T13:07:01.439291Z"
        },
        "id": "CkgrsDq5KKA8"
      },
      "outputs": [],
      "execution_count": null
    },
    {
      "cell_type": "code",
      "source": [
        "print(accuracy)\n",
        "print(roc_auc)\n",
        "print(cv_score)"
      ],
      "metadata": {
        "trusted": true,
        "execution": {
          "iopub.status.busy": "2026-05-30T13:07:31.61868Z",
          "iopub.execute_input": "2026-05-30T13:07:31.619188Z",
          "iopub.status.idle": "2026-05-30T13:07:31.624737Z",
          "shell.execute_reply.started": "2026-05-30T13:07:31.619156Z",
          "shell.execute_reply": "2026-05-30T13:07:31.62357Z"
        },
        "id": "IVCcY5BnKKA8"
      },
      "outputs": [],
      "execution_count": null
    },
    {
      "cell_type": "code",
      "source": [
        "model=pipeline.named_steps[\"model\"]"
      ],
      "metadata": {
        "trusted": true,
        "execution": {
          "iopub.status.busy": "2026-05-30T13:07:34.424406Z",
          "iopub.execute_input": "2026-05-30T13:07:34.424925Z",
          "iopub.status.idle": "2026-05-30T13:07:34.429668Z",
          "shell.execute_reply.started": "2026-05-30T13:07:34.424882Z",
          "shell.execute_reply": "2026-05-30T13:07:34.428486Z"
        },
        "id": "xCrudSLsKKA9"
      },
      "outputs": [],
      "execution_count": null
    },
    {
      "cell_type": "code",
      "source": [
        "encoded_cat_features = pipeline.named_steps[\n",
        "'preprocessor'].transformers_[1][1].named_steps['encoder'].get_feature_names_out(categorical)"
      ],
      "metadata": {
        "trusted": true,
        "execution": {
          "iopub.status.busy": "2026-05-30T13:07:36.693929Z",
          "iopub.execute_input": "2026-05-30T13:07:36.694884Z",
          "iopub.status.idle": "2026-05-30T13:07:36.700005Z",
          "shell.execute_reply.started": "2026-05-30T13:07:36.69481Z",
          "shell.execute_reply": "2026-05-30T13:07:36.699015Z"
        },
        "id": "s95WkpQjKKA9"
      },
      "outputs": [],
      "execution_count": null
    },
    {
      "cell_type": "code",
      "source": [
        "all_features = np.concatenate([\n",
        "numerical,encoded_cat_features])"
      ],
      "metadata": {
        "trusted": true,
        "execution": {
          "iopub.status.busy": "2026-05-30T13:07:39.225478Z",
          "iopub.execute_input": "2026-05-30T13:07:39.226497Z",
          "iopub.status.idle": "2026-05-30T13:07:39.231324Z",
          "shell.execute_reply.started": "2026-05-30T13:07:39.22646Z",
          "shell.execute_reply": "2026-05-30T13:07:39.229965Z"
        },
        "id": "Hng34lhrKKA9"
      },
      "outputs": [],
      "execution_count": null
    },
    {
      "cell_type": "code",
      "source": [
        "importance=model.feature_importances_\n",
        "feature_importance_df=pd.DataFrame({\"feature\":all_features,\"Importance\":importance})"
      ],
      "metadata": {
        "trusted": true,
        "execution": {
          "iopub.status.busy": "2026-05-30T13:07:41.783975Z",
          "iopub.execute_input": "2026-05-30T13:07:41.784573Z",
          "iopub.status.idle": "2026-05-30T13:07:41.791943Z",
          "shell.execute_reply.started": "2026-05-30T13:07:41.784541Z",
          "shell.execute_reply": "2026-05-30T13:07:41.790764Z"
        },
        "id": "xy3dzVUXKKA9"
      },
      "outputs": [],
      "execution_count": null
    },
    {
      "cell_type": "code",
      "source": [
        "feature_importance_df = feature_importance_df.sort_values(\n",
        "by=\"Importance\",ascending=False)"
      ],
      "metadata": {
        "trusted": true,
        "execution": {
          "iopub.status.busy": "2026-05-30T13:07:52.585169Z",
          "iopub.execute_input": "2026-05-30T13:07:52.585628Z",
          "iopub.status.idle": "2026-05-30T13:07:52.591378Z",
          "shell.execute_reply.started": "2026-05-30T13:07:52.585599Z",
          "shell.execute_reply": "2026-05-30T13:07:52.59043Z"
        },
        "id": "wmr8o9b7KKBE"
      },
      "outputs": [],
      "execution_count": null
    },
    {
      "cell_type": "code",
      "source": [
        "print(feature_importance_df)"
      ],
      "metadata": {
        "trusted": true,
        "execution": {
          "iopub.status.busy": "2026-05-30T09:52:46.213845Z",
          "iopub.execute_input": "2026-05-30T09:52:46.214158Z",
          "iopub.status.idle": "2026-05-30T09:52:46.22037Z",
          "shell.execute_reply.started": "2026-05-30T09:52:46.214135Z",
          "shell.execute_reply": "2026-05-30T09:52:46.21942Z"
        },
        "id": "hXBAmQwUKKBF"
      },
      "outputs": [],
      "execution_count": null
    },
    {
      "cell_type": "code",
      "source": [
        "param_distributions = {\n",
        " \"model__n_estimators\": [100, 200, 300, 500],\n",
        " \"model__max_depth\": [5,10,20,25],\n",
        " \"model__min_samples_split\": [2, 5, 10],\n",
        "\n",
        " \"model__min_samples_leaf\": [1, 2, 4],\n",
        "\n",
        "}\n",
        "\n",
        "param_distributions"
      ],
      "metadata": {
        "trusted": true,
        "execution": {
          "iopub.status.busy": "2026-05-30T13:08:03.673732Z",
          "iopub.execute_input": "2026-05-30T13:08:03.674143Z",
          "iopub.status.idle": "2026-05-30T13:08:03.681669Z",
          "shell.execute_reply.started": "2026-05-30T13:08:03.674109Z",
          "shell.execute_reply": "2026-05-30T13:08:03.680593Z"
        },
        "id": "XlI5drj8KKBG"
      },
      "outputs": [],
      "execution_count": null
    },
    {
      "cell_type": "code",
      "source": [
        "random_search = RandomizedSearchCV(estimator=pipeline,\n",
        "param_distributions=param_distributions,\n",
        "n_iter=20,scoring='roc_auc',\n",
        "cv=5,verbose=2,random_state=42,\n",
        "n_jobs=-1)\n",
        "random_search"
      ],
      "metadata": {
        "trusted": true,
        "execution": {
          "iopub.status.busy": "2026-05-30T13:08:09.134603Z",
          "iopub.execute_input": "2026-05-30T13:08:09.134962Z",
          "iopub.status.idle": "2026-05-30T13:08:09.317519Z",
          "shell.execute_reply.started": "2026-05-30T13:08:09.134931Z",
          "shell.execute_reply": "2026-05-30T13:08:09.316455Z"
        },
        "id": "4pq7ih6kKKBG"
      },
      "outputs": [],
      "execution_count": null
    },
    {
      "cell_type": "code",
      "source": [
        "random_search.fit(X_train,y_train)"
      ],
      "metadata": {
        "trusted": true,
        "execution": {
          "iopub.status.busy": "2026-05-30T13:08:39.964819Z",
          "iopub.execute_input": "2026-05-30T13:08:39.965784Z",
          "iopub.status.idle": "2026-05-30T13:11:26.021347Z",
          "shell.execute_reply.started": "2026-05-30T13:08:39.965751Z",
          "shell.execute_reply": "2026-05-30T13:11:26.020231Z"
        },
        "id": "ehBMaRpQKKBH"
      },
      "outputs": [],
      "execution_count": null
    },
    {
      "cell_type": "code",
      "source": [
        "random_search.best_params_"
      ],
      "metadata": {
        "trusted": true,
        "execution": {
          "iopub.status.busy": "2026-05-30T13:12:39.744088Z",
          "iopub.execute_input": "2026-05-30T13:12:39.744592Z",
          "iopub.status.idle": "2026-05-30T13:12:39.752462Z",
          "shell.execute_reply.started": "2026-05-30T13:12:39.744534Z",
          "shell.execute_reply": "2026-05-30T13:12:39.751214Z"
        },
        "id": "fee2gvQ-KKBH"
      },
      "outputs": [],
      "execution_count": null
    },
    {
      "cell_type": "code",
      "source": [
        "best_model=random_search.best_estimator_\n"
      ],
      "metadata": {
        "trusted": true,
        "execution": {
          "iopub.status.busy": "2026-05-30T13:12:44.133589Z",
          "iopub.execute_input": "2026-05-30T13:12:44.134035Z",
          "iopub.status.idle": "2026-05-30T13:12:44.139425Z",
          "shell.execute_reply.started": "2026-05-30T13:12:44.134004Z",
          "shell.execute_reply": "2026-05-30T13:12:44.138238Z"
        },
        "id": "svco7ibTKKBI"
      },
      "outputs": [],
      "execution_count": null
    },
    {
      "cell_type": "code",
      "source": [
        "y_pred_best=best_model.predict(X_test)\n",
        "y_pred[:5]"
      ],
      "metadata": {
        "trusted": true,
        "execution": {
          "iopub.status.busy": "2026-05-30T13:13:18.72375Z",
          "iopub.execute_input": "2026-05-30T13:13:18.724108Z",
          "iopub.status.idle": "2026-05-30T13:13:18.749281Z",
          "shell.execute_reply.started": "2026-05-30T13:13:18.724081Z",
          "shell.execute_reply": "2026-05-30T13:13:18.74824Z"
        },
        "id": "OOwO68ldKKBI"
      },
      "outputs": [],
      "execution_count": null
    },
    {
      "cell_type": "code",
      "source": [
        "import shap\n",
        "shap.initjs"
      ],
      "metadata": {
        "trusted": true,
        "execution": {
          "iopub.status.busy": "2026-05-30T13:14:08.593822Z",
          "iopub.execute_input": "2026-05-30T13:14:08.594467Z",
          "iopub.status.idle": "2026-05-30T13:14:12.937101Z",
          "shell.execute_reply.started": "2026-05-30T13:14:08.594435Z",
          "shell.execute_reply": "2026-05-30T13:14:12.936079Z"
        },
        "id": "0AKCt91jKKBJ"
      },
      "outputs": [],
      "execution_count": null
    },
    {
      "cell_type": "code",
      "source": [
        "best_model_=best_model.named_steps['model']\n",
        "best_model_"
      ],
      "metadata": {
        "trusted": true,
        "execution": {
          "iopub.status.busy": "2026-05-30T13:15:53.033732Z",
          "iopub.execute_input": "2026-05-30T13:15:53.034174Z",
          "iopub.status.idle": "2026-05-30T13:15:53.042422Z",
          "shell.execute_reply.started": "2026-05-30T13:15:53.034142Z",
          "shell.execute_reply": "2026-05-30T13:15:53.041317Z"
        },
        "id": "Y_fVAiPSKKBJ"
      },
      "outputs": [],
      "execution_count": null
    },
    {
      "cell_type": "code",
      "source": [
        "X_train_processed = best_model.named_steps['preprocessor'].transform(X_train)\n",
        "X_test_processed = best_model.named_steps['preprocessor'].transform(X_test)\n",
        "X_train_processed.shape"
      ],
      "metadata": {
        "trusted": true,
        "execution": {
          "iopub.status.busy": "2026-05-30T13:18:00.49028Z",
          "iopub.execute_input": "2026-05-30T13:18:00.491355Z",
          "iopub.status.idle": "2026-05-30T13:18:00.523311Z",
          "shell.execute_reply.started": "2026-05-30T13:18:00.491318Z",
          "shell.execute_reply": "2026-05-30T13:18:00.521977Z"
        },
        "id": "_wup1mQGKKBK"
      },
      "outputs": [],
      "execution_count": null
    },
    {
      "cell_type": "code",
      "source": [
        "encoded_cat_features = best_model.named_steps['preprocessor'].transformers_[1][1].named_steps[\n",
        "'encoder'].get_feature_names_out(categorical)\n",
        "all_features = np.concatenate([numerical,encoded_cat_features])"
      ],
      "metadata": {
        "trusted": true,
        "execution": {
          "iopub.status.busy": "2026-05-30T13:19:08.693924Z",
          "iopub.execute_input": "2026-05-30T13:19:08.694408Z",
          "iopub.status.idle": "2026-05-30T13:19:08.700052Z",
          "shell.execute_reply.started": "2026-05-30T13:19:08.694378Z",
          "shell.execute_reply": "2026-05-30T13:19:08.698888Z"
        },
        "id": "NydXAZfHKKBK"
      },
      "outputs": [],
      "execution_count": null
    },
    {
      "cell_type": "code",
      "source": [
        "explainer=shap.TreeExplainer(best_model_)\n",
        "explainer"
      ],
      "metadata": {
        "trusted": true,
        "execution": {
          "iopub.status.busy": "2026-05-30T13:20:14.753607Z",
          "iopub.execute_input": "2026-05-30T13:20:14.754595Z",
          "iopub.status.idle": "2026-05-30T13:20:14.76935Z",
          "shell.execute_reply.started": "2026-05-30T13:20:14.75456Z",
          "shell.execute_reply": "2026-05-30T13:20:14.768405Z"
        },
        "id": "MvLZzKR3KKBK"
      },
      "outputs": [],
      "execution_count": null
    },
    {
      "cell_type": "code",
      "source": [
        "shap_values = explainer.shap_values(X_test_processed)"
      ],
      "metadata": {
        "trusted": true,
        "execution": {
          "iopub.status.busy": "2026-05-30T13:20:32.013819Z",
          "iopub.execute_input": "2026-05-30T13:20:32.014823Z",
          "iopub.status.idle": "2026-05-30T13:20:37.206414Z",
          "shell.execute_reply.started": "2026-05-30T13:20:32.014774Z",
          "shell.execute_reply": "2026-05-30T13:20:37.205332Z"
        },
        "id": "_DFHzLwdKKBL"
      },
      "outputs": [],
      "execution_count": null
    },
    {
      "cell_type": "code",
      "source": [
        "shap.summary_plot(shap_values[::1],X_test_processed,feature_names=all_features)"
      ],
      "metadata": {
        "trusted": true,
        "execution": {
          "iopub.status.busy": "2026-05-30T13:21:30.983305Z",
          "iopub.execute_input": "2026-05-30T13:21:30.983606Z",
          "iopub.status.idle": "2026-05-30T13:21:31.477906Z",
          "shell.execute_reply.started": "2026-05-30T13:21:30.983583Z",
          "shell.execute_reply": "2026-05-30T13:21:31.477016Z"
        },
        "id": "80dmaPMLKKBM"
      },
      "outputs": [],
      "execution_count": null
    },
    {
      "cell_type": "code",
      "source": [
        "shap.summary_plot(shap_values,X_test_processed,feature_names=all_features,plot_type=\"bar\")\n",
        "\n",
        "sample_index = 0\n",
        "shap.force_plot(explainer.expected_value[1],shap_values[sample_index::1],X_test_processed[sample_index],feature_names=all_features)"
      ],
      "metadata": {
        "trusted": true,
        "execution": {
          "iopub.status.busy": "2026-05-30T13:28:18.843934Z",
          "iopub.execute_input": "2026-05-30T13:28:18.84488Z",
          "iopub.status.idle": "2026-05-30T13:28:19.02894Z",
          "shell.execute_reply.started": "2026-05-30T13:28:18.844812Z",
          "shell.execute_reply": "2026-05-30T13:28:19.027573Z"
        },
        "id": "zXccLBihKKBM"
      },
      "outputs": [],
      "execution_count": null
    }
  ]
}