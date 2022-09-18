FROM nvidia/cuda:11.7.1-base-ubuntu22.04

WORKDIR /app

# set bash as current shell
RUN chsh -s /bin/bash
SHELL ["/bin/bash", "-c"]

# install anaconda
RUN apt-get update
RUN apt-get install -y wget bzip2 ca-certificates libglib2.0-0 libxext6 libsm6 libxrender1 git mercurial subversion && \
    apt-get clean
RUN wget https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh
RUN /bin/bash Anaconda3-2022.05-Linux-x86_64.sh -b -p $HOME/anaconda3
RUN echo 'export PATH=/root/anaconda3/bin:$PATH' >> /root/.bashrc 

RUN . /root/.bashrc && \
    /root/anaconda3/bin/conda init bash && \
    /root/anaconda3/bin/conda create -n gaitor python=3.9 anaconda && \
    /root/anaconda3/bin/conda activate gaitor

RUN conda install --file /tmp/requirements.yaml

COPY . .

# Download ML Models (Initialization)
RUN python init.py

ENTRYPOINT [ "python3" , "main.py"]