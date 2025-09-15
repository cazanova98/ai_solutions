import datetime
import sagemaker
from sagemaker.huggingface import HuggingFace
from sagemaker.serverless import ServerlessInferenceConfig
from dotenv import load_dotenv

# AWS role with SageMaker permissions
load_dotenv()
role = "arn:aws:iam::<account_id>:role/service-role/AmazonSageMaker-ExecutionRole-XXXX"

date_str = datetime.datetime.now().strftime("%Y-%m-%d")
output_path = f"s3://dolphintech-bucket/fraud_checks/models/{date_str}/"

# HuggingFace estimator
huggingface_estimator = HuggingFace(
    entry_point='train_llama.py',   # your script
    source_dir='.',                 # folder with requirements.txt
    instance_type='ml.p3.2xlarge',  # GPU instance
    instance_count=1,
    role=role,
    transformers_version='4.33',
    pytorch_version='2.1',
    py_version='py311',
    hyperparameters={
        'epochs': 2,
        'batch_size': 2
    }
)

# Launch training (dataset can be local or S3)
huggingface_estimator.fit({
    "train": "s3://dolphintech-bucket/fraud_checks/"
}, wait=True, job_name=f"fraud-detection-{date_str}", output_path=output_path)

print("Trained model is at:", huggingface_estimator.model_data) #To see exact S3 path of trained model

# Deploy to SageMaker serverless endpoint
serverless_config = ServerlessInferenceConfig(
    memory_size_in_mb=4096,  # This is a fair amount of memory for a model like LLaMA
    max_concurrency=50        # Adjust this as the business needs
)
predictor = huggingface_estimator.deploy(
    serverless_inference_config=serverless_config,
    endpoint_name='check-fraud-detection-endpoint-serverless'
)

# Example prediction
data = {"text": "Sample check data to classify as fraud/not-fraud"}
prediction = predictor.predict(data)
print(prediction)