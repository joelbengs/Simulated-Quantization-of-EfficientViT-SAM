# Author: Joel Bengs 2024-06-12                                                                                                                     Modified  # Author Joel Bengs 2024-06-12
# Project run in pytorch 2.0.1, but this is not available as base image
FROM pytorch/pytorch:latest

# Install git.
RUN apt-get update && apt-get install -y git

RUN git clone https://oauth2:glpat-94tD9SMe3jMTBzrgUypz@gitlab.rosetta.ericssondevops.com/joel.bengs/thesis-test-repo /workspace/thesis-test-repo

WORKDIR /workspace/thesis-test-repo

RUN ls -la

RUN pip install -r requirements.txt

# Each Run commmand creates an intermediate container, which is only removed if the command succeeds. Else it is left behind.