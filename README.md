# Semantic Search on Images from PDF documents

## Overview
Semantic search is an approach to retrieving information based on understanding the contextual meaning of search terms. Many organizations have repositories of PDF documents containing millions of images. Providing easy access to these images as a searchable knowledge base for their teams can be essential to their business. However, implementing accurate semantic search capabilities for images can be complex, time-consuming, and expensive to build.

This repository contains the cloudformation template that builds an application to retrieve images from PDFs and build a Enterprise SEarch Application powered by Generative AI to retrieve relevant images and metadata. The solution extracts images from PDF documents. The images are analyzed using the machine learning (ML) capabilities of Amazon Rekognition. Rekognition Image is an image recognition service that analysis images to detect objects, scenes, activities, landmarks, faces, dominant colors, and image quality. Rekognition Image lets you easily build powerful applications to search, verify, and organize millions of images.
This code repo uses [streamlit](https://docs.streamlit.io/)

## Code Layout

| Path              | Description                                                                             |
|:------------------|:----------------------------------------------------------------------------------------|
| main.py           | 	Contains function that supports streamlit operations                                   |
| utils.py          | 	Contains functions that support creation of OpenSearch Collections and access policies |
| cfn_template.yaml | 	CloudFormation template to deploy the solution                                         |
| notebook          | 	Jupyter Notebook containing the code for the entire CS02 workflow                      |


## Pre-requisites

1. [Install AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html)
2. [Setup AWS profile](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-configure.html)
3. [Install NPM, node package managers](https://docs.npmjs.com/downloading-and-installing-node-js-and-npm)

## Deploy using CFN Template

The below instructions deploys the Streamlit app into an EC2 instance.

1. Deploy the `cfn_template.yaml` into your account.
2. Navigate to the EC2 instance created by the cloudformation template. (Please refer to the CFN Stack **output** to find the EC2 instance id) 
3. Lookup the instance id in the [EC2 console](https://console.aws.amazon.com/ec2)
4. Click on the instance id, click **Connect** on the top menu and choose **Session Manager** to connect and open the EC2 terminal 
5. [OPTIONAL] Login as root (`sudo su`)
6. Then check the /var/log/cloud-init-output.log to see if the initialization scripts have successfully run (tail -100f /var/log/cloud-init-output.log)
7. Login as ec2-user and navigate to `/home/ec2-user/` directory (`sudo -i -u ec2-user`) 
8. Run `streamlit run main.py` to run the streamlit application
9. The following prompt should appear upon execution of the previous command, please open the **External URL** in your browser to view the web application
You can now view your Streamlit app in your browser.

Network URL: http://xxx.xx.xx.xxx:8501

External URL: http://x.xx.xx.xx:8501
   


