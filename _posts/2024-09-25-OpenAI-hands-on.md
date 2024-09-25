---
layout: post
title: OpenAI Hands-on
subtitle: People are usually familiar with supervised or unsupervised learning but not Reinforcement Learning. In RL, we choose an action using a policy and given some data to maximize the expected long-term reward (all these terms will be explained).
cover-img: assets/img/posts/2024-07-9/cover.JPG
thumbnail-img: /assets/img/posts/2024-07-9/rl.png
share-img: assets/img/posts/2024-07-9/cover.JPG
tags: [AI, ML, Reinforcement Learning]
author: Fares Ben Slimane
date:   2024-07-9
---

## Context and Plan

This is a quick post for developers who want to use OpenAI API for their projects or businesses. We will be using OpenAI API for chat completion and image generation for a web app for TikTokers since there is a chance that I am going to be one in the near future (I know shocking!). 

The task of creating thumbnails or text descriptions can be time-consuming, especially for busy people like influencers, so why not create a web extension that can help with that? The user will pass keywords onto the app to get a full detailed description and then a nice-looking thumbnail image. 



## Prepare Environment and platform

You need first to create an account in Openai and create an API key (https://platform.openai.com/). The access key is to access the openAI features, and should not be shared with others, that's why we usually want to save it in an env to avoid exposing it in our frontend application. You can get free credits in the beginning, but later on, you need to have a subscription for as little as 20$/month. 

We can use NPM (node package manager) to manage and install useful libraries like openAI, dotenv. Navigate to your root folder and perform the following commands:

npm init //Creates package.json
npm install openai dotenv //Installs necessary libraries

Save you openAI token in .env file, like this:

OPENAI_TOKEN=yourkey

Set up your Open AI configuration using your access key:
'
//Set OpenAI configuration
const{ Configuration, OpenAIApi } = require('openai');

//Get env saved token
require('dotenv').config()

const config = new Configuration(
    {appiKey: process.env.OPENAI_TOKEN}
)

const openai = new OpenAIApi(config);
module ex
'

