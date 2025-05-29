FROM python:3.11

COPY ./configuration /configuration

COPY ./steps /steps

COPY ./utils /utils

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt














# FROM python:3.9

# WORKDIR /usr/local/src

# # Create Directory

# RUN mkdir kfp

# WORKDIR /usr/local/src/kfp

# RUN mkdir components

# WORKDIR /usr/local/src/kfp/components

# # Copy the current directory contents into the container at /utils
# COPY ./utils/ /utils/

# COPY ./configuration/ /configuration/

# COPY ./steps/ steps/

# COPY requirements.txt .

# RUN pip install --no-cache-dir -r requirements.txt