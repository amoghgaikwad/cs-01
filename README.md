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


## Bedrock Setup - Add model access

You can add access to a model in Amazon Bedrock with the following steps:

1. Open the account console and change the region to **us-east-1**.
2. Open the Amazon Bedrock console at [Amazon Bedrock console](https://console.aws.amazon.com/bedrock).
3. Go to the **Model access** link on the left side navigation panel in Amazon Bedrock, or go to the Edit model access page.
4. Select the checkbox next to the `Titan Embeddings G1 -Text`, `Anthropic - Claude and Claude Instant` models.
5. Select **Request Model Access** to add access to any third party models through Amazon Marketplace. 
   >**Note**: Your use of Amazon Bedrock and it's models is subject to the seller's pricing terms, EULA and the Amazon Bedrock service terms.
6. Select the **Save Changes** button in the lower right corner of the page. It may take several minutes to save changes to the Model access page.
7. Models will show as **Available** on the Model access page under Access status, if access is granted.

Some users may not have IAM permissions to add access to models. A banner message will appear if you try to add access to models, and you are a non-admin user on your account. You will need to contact your account administrator to request that they add access to the models before you can use them in Amazon Bedrock.

When you have access to the model, you can select it using the View model access button or the Edit model access page. Accounts do not have access to Amazon models by default.

## Deploy using CFN Template

The below instructions deploys the Streamlit app into an EC2 instance.

1. Deploy the `cfn_template.yaml` into your account, preferably in us-east-1 or in one of [supported regions](https://docs.aws.amazon.com/bedrock/latest/userguide/what-is-bedrock.html#bedrock-regions).
2. Navigate to the EC2 instance created by the cloudformation template. (Please refer to the CFN Stack **output** to find the EC2 instance id) 
3. Lookup the instance id in the [EC2 console](https://console.aws.amazon.com/ec2)
4. Click on the instance id, click **Connect** on the top menu and choose **Session Manager** to connect and open the EC2 terminal 
5. [OPTIONAL] Login as root 
   > `sudo su`
6. Then check the /var/log/cloud-init-output.log to see if the initialization scripts have successfully run (tail -100f /var/log/cloud-init-output.log)
7. Run the following command to login as `ec2-user` and navigate to `/home/ec2-user/` directory.
    > `sudo -i -u ec2-user`
8. Run `streamlit run main.py` to run the streamlit application
9. The following prompt should appear upon execution of the previous command, please open the **External URL** in your browser to view the web application
You can now view your Streamlit app in your browser.

Network URL: http://xxx.xx.xx.xxx:8501

External URL: http://x.xx.xx.xx:8501
   


