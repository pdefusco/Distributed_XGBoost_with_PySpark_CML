# Distributed XGBoost with PySpark in CML

## Objective

This GitHub repository provides notebook examples for using Spark to distribute XGBoost applications in Cloudera Machine Learning.

## About Cloudera Machine Learning

Cloudera Machine Learning (CML) on Cloudera Data Platform accelerates time-to-value by enabling data scientists to collaborate in a single unified platform that is all inclusive for powering any AI use case. Purpose-built for agile experimentation and production ML workflows, Cloudera Machine Learning manages everything from data preparation to MLOps, to predictive reporting. Solve mission critical ML challenges along the entire lifecycle with greater speed and agility to discover opportunities which can mean the difference for your business.

## Requirements

In order to reproduce this project you need the following:

* A CML Workspace in Private or Public Cloud (AWS, Azure, OCP and Cloudera ECS OK).
* Basic familiarity with Python, Spark, Git and Machine Learning is recommended.
* The code in the notebooks only requires basic modifications. Instructions for these are provided in the "Running the Notebooks" section.
* The CML Workspace does not require a Custom Runtime. The "Standard Runtime" provided in every workspace will be ok.

## Project Setup

Create a CML Project by cloning this Git repository from the CML UI.

Launch a CML Session with the following configurations:

```
Editor: JupyterLab
Kernel: Python 3.8 or above
Edition: Standard
Version: any version ok
Enable Spark: Spark 3.2.0 and above ok
Resource Profile: 2vCPU / 4GiB Memory - 0 GPU
```

Open the Terminal and install the required packages contained in the "requirements.txt" file with the following command:

```
pip3 install -r requirements.txt
```

![alt text](img/cml_terminal.png)

![alt text](img/cml_terminal_2.png)

Navigate to the Workspace "Site Administration" page, open the "Runtime" tab and ensure that the "Enable CPU Bursting" option is enabled.

![alt text](img/site_admin.png)

## Running the Notebooks



## Summary and Next Steps
