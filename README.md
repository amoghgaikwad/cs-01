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

## Deploy


   


