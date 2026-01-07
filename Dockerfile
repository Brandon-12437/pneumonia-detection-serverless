FROM public.ecr.aws/lambda/python:3.11

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy model and code
COPY xception_pneumonia.onnx .
COPY handler.py .

# Command for Lambda to call your function
CMD [ "handler.lambda_handler" ]









