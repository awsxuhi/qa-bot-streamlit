# Comments from xuhi@amajzon.com.
# Given I am building images on my M1 MacOs. The option of '--platform=linux/amd64'. 
# No need to put "architecture: _lambda.Architecture.X86_64" in your CDK DockerImageFunction, because the default value is X86_64.
FROM --platform=linux/amd64 python:3.8-slim

# Create /app directory
RUN mkdir -p /app

# Copy requirements.txt
COPY requirements.txt /app
RUN ls -l /app

# Install the specified packages
# RUN pip install -r /app/requirements.txt --target /app
RUN pip install -r /app/requirements.txt

# Copy function code, leverage the cache during re-build, so put this line below.
COPY webapp.py /app/webapp.py

# Set the CMD to your webapp.py
CMD ["streamlit", "run", "/app/webapp.py"]
