import requests
import json

# This specific path is required by the Lambda Emulator
url = "http://localhost:9000/2015-03-31/functions/function/invocations"

payload = {
    "url": "https://pneumonia-xray-images.s3.us-east-1.amazonaws.com/journal.pone.0256630.g001.PNG"
}

try:
    print("Testing local Lambda container...")
    response = requests.post(url, json=payload)
    
    # Lambda RIE returns a JSON with 'statusCode' and 'body'
    raw_result = response.json()
    
    # We must parse the 'body' string back into a dictionary
    predictions = json.loads(raw_result["body"])

    print("\n" + "="*50)
    print("ANALYSIS REPORT")
    print("-" * 50)
    print(f"Normal:    {predictions.get('normal', 0):.2%}")
    print(f"Pneumonia: {predictions.get('pneumonia', 0):.2%}")
    print("-" * 50)

    # Medical Guidance based on Output
    if predictions.get('pneumonia', 0) > 0.5:
        print("RESULT: High probability of pneumonia detected.")
        print("\nMEDICAL ADVICE:")
        print("1. Seek immediate consultation with a Radiologist or GP.")
        print("2. Do not start medication without a formal prescription.")
        print("3. Clinical correlation (fever, cough, history) is required.")
    else:
        print("RESULT: No significant signs of pneumonia detected.")
        print("\nMEDICAL ADVICE:")
        print("1. If you have a persistent cough or chest pain, see a doctor.")
        print("2. Early infections may not appear immediately on X-rays.")
    
    print("\nNOTICE: For research purposes only. Not a medical diagnosis.")
    print("="*50 + "\n")

except Exception as e:
    print(f"Error connecting to Docker: {e}")
    print("Check if the container is running on port 9000.")