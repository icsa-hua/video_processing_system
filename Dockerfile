# Use an official Python runtime as the base image
FROM python:3.10.13

# Set the working directory in the container
WORKDIR /app

# Copy your project files into the container
COPY requirements.txt ./

# Install any project-specific dependencies
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

# Bundle app source
COPY . ./app

EXPOSE 3333

ENTRYPOINT ["python3"]

# Specify the command to run your project
CMD [ "./app/test_dummy_pipeline.py", "--source", "../samples/10_DrivingWith.mp4","--show", "--verbose"]