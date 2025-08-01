# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire openai2civitai directory into the container at /app/openai2civitai
COPY openai2civitai/ /app/openai2civitai/

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Run server.py when the container launches
CMD ["uvicorn", "openai2civitai.server:app", "--host", "0.0.0.0", "--port", "8000"]
