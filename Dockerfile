# Author Joel Bengs 2024-06-12
# Project run in pytorch 2.0.1, but this is not available as base image
FROM pytorch/pytorch:latest

# Install git
RUN apt-get update && apt-get install -y git

# RUN git clone https://gitlab.rosetta.ericssondevops.com/joel.bengs/thesis-test-repo
# The above doesn't work due to credentials. You can create a personal access token in Gitab and then:
RUN git clone https://oauth2:<your_token>@gitlab.rosetta.ericssondevops.com/joel.bengs/thesis-test-repo

WORKDIR /thesis-test-repo

RUN pip install -r requirements.txt
