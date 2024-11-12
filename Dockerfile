# Use an official Python base image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt /app/

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the source code
COPY . /app/

# Expose ports for the application (8888 is used for Jupyter Notebook, if needed)
EXPOSE 8888

# Run the main script
CMD ["python", "train_docker.py"]
