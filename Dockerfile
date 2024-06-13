# Author: Joel Bengs 2024-06-12

# NOTE! This Dockerfile needs uncommenting to function. Read it carefully
# Each Run commmand creates an intermediate container, which is only removed if the command succeeds. Else it is left behind.

# A good base-image is the following, which includes PyTorch, CUDA 11.3, and cuDNN 8.2.
FROM nvcr.io/nvidia/pytorch:21.09-py3

# Install git.
RUN apt-get update && apt-get install -y git

WORKDIR /workspace/thesis-test-repo

# Here, you must sort the credentials to the repo. One way is to create an oauth2 token and insert it. The last /workspace/thesis-test-repo specifies location
# RUN git clone https://oauth2:glpat-94tD9SMe3jMTBzrgUypz@gitlab.rosetta.ericssondevops.com/joel.bengs/thesis-test-repo /workspace/thesis-test-repo
# Another way is to cpy over the SSH keys into the image, and then use ssh to clone
# RUN git clone <ssh link to repo>

# After successful cloning, isntall the requirements either manually or by uncommenting below
# RUN pip install -r requirements.txt