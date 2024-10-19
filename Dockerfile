# Use an official Python runtime as a parent image
FROM python:3.12.3-slim

# Install required system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libjpeg-dev \
    libmagic-dev \
    zlib1g-dev \
    libopenjp2-7-dev \
    libpng-dev \
    libpoppler-cpp-dev \
    pkg-config \
    gcc \
    git \
    libqpdf-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /ragbuilder

# Copy the current directory contents into the container at /ragbuilder
COPY . /ragbuilder

 
# Install pip and upgrade setuptools and wheel
RUN pip install --upgrade pip setuptools wheel

# RUN pip install --use-pep517 -r requirements.txt
# RUN python3 -m build 
RUN pip install dist/*.gz
# Delete all files in the current directory
WORKDIR /

# Optionally, you can delete hidden files and directories as well
RUN rm -rf /ragbuilder/*

WORKDIR /ragbuilder

COPY LICENSE /ragbuilder/LICENSE
COPY .env-Sample /ragbuilder/.env-Sample
COPY README.md /ragbuilder/README.md



# Make port 80 available to the world outside this container
EXPOSE 8005


# Run app.py when the container launches
CMD ["ragbuilder"]
