#!/usr/bin/env python3
"""
AWS SageMaker Job Launcher for Advanced Football Pass Prediction

This script launches training jobs on SageMaker with proper configuration,
data handling, and model deployment capabilities.
"""

import os
import json
import boto3
import sagemaker
from sagemaker.pytorch import PyTorch
from sagemaker.inputs import TrainingInput
from sagemaker import get_execution_role
from datetime import datetime
import argparse
from pathlib import Path
import tarfile


class SageMakerJobLauncher:
    """Manages SageMaker training jobs for the advanced pass prediction model"""
    
    def __init__(self, region='us-west-2', role=None):
        """Initialize SageMaker session and configuration"""
        self.session = sagemaker.Session()
        self.region = region
        self.role = role or get_execution_role()
        self.bucket = self.session.default_bucket()
        
        print(f"SageMaker session initialized")
        print(f"Region: {self.region}")
        print(f"Role: {self.role}")
        print(f"Default bucket: {self.bucket}")
    
    def upload_data_to_s3(self, local_data_path, s3_prefix='football-pass-prediction/data'):
        """Upload training data to S3"""
        print(f"Uploading data from {local_data_path} to S3...")
        
        local_path = Path(local_data_path)
        if not local_path.exists():
            raise FileNotFoundError(f"Data path {local_data_path} does not exist")
        
        # Create a tar.gz file if it's a directory
        if local_path.is_dir():
            tar_path = local_path.parent / f"{local_path.name}.tar.gz"
            with tarfile.open(tar_path, "w:gz") as tar:
                tar.add(local_path, arcname=local_path.name)
            upload_path = tar_path
        else:
            upload_path = local_path
        
        # Upload to S3
        s3_data_path = self.session.upload_data(
            path=str(upload_path),
            bucket=self.bucket,
            key_prefix=s3_prefix
        )
        
        print(f"Data uploaded to: {s3_data_path}")
        return s3_data_path
    
    def upload_source_code(self, source_dir='./'):
        """Upload source code to S3"""
        print("Preparing source code...")
        
        source_path = Path(source_dir)
        
        # Create source code tar
        code_tar = "source.tar.gz"
        with tarfile.open(code_tar, "w:gz") as tar:
            # Add the training script and dependencies
            tar.add(source_path / "advanced_pass_prediction.py", arcname="advanced_pass_prediction.py")
            tar.add(source_path / "sagemaker_train.py", arcname="sagemaker_train.py")
            if (source_path / "requirements.txt").exists():
                tar.add(source_path / "requirements.txt", arcname="requirements.txt")
        
        # Upload to S3
        s3_code_path = self.session.upload_data(
            path=code_tar,
            bucket=self.bucket,
            key_prefix='football-pass-prediction/code'
        )
        
        # Cleanup
        os.remove(code_tar)
        
        print(f"Source code uploaded to: {s3_code_path}")
        return s3_code_path
    
    def create_training_job(self, 
                          s3_data_path, 
                          job_name=None,
                          instance_type='ml.p3.2xlarge',
                          instance_count=1,
                          hyperparameters=None,
                          max_run_time=3600):
        """Create and launch SageMaker training job"""
        
        if job_name is None:
            timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            job_name = f'football-pass-prediction-{timestamp}'
        
        # Default hyperparameters
        default_hyperparameters = {
            'epochs': 20,
            'batch-size': 64,
            'learning-rate': 1e-4,
            'weight-decay': 1e-5,
            'window': 12,
            'context-window': 24,
            'd-model': 256,
            'n-heads': 8,
            'n-layers': 6,
            'dropout': 0.1,
            'csv-filename': 'data.csv'  # Default CSV filename
        }
        
        if hyperparameters:
            default_hyperparameters.update(hyperparameters)
        
        print(f"Creating training job: {job_name}")
        print(f"Instance type: {instance_type}")
        print(f"Instance count: {instance_count}")
        print(f"Hyperparameters: {json.dumps(default_hyperparameters, indent=2)}")
        
        # Create PyTorch estimator
        estimator = PyTorch(
            entry_point='sagemaker_train.py',
            source_dir='.',
            role=self.role,
            instance_count=instance_count,
            instance_type=instance_type,
            framework_version='1.12.0',
            py_version='py38',
            hyperparameters=default_hyperparameters,
            max_run=max_run_time,
            enable_sagemaker_metrics=True,
            metric_definitions=[
                {'Name': 'train:accuracy', 'Regex': 'Train Acc: ([0-9\\.]+)'},
                {'Name': 'validation:accuracy', 'Regex': 'Val Acc: ([0-9\\.]+)'},
                {'Name': 'validation:top3_accuracy', 'Regex': 'Top-3 Acc: ([0-9\\.]+)'},
                {'Name': 'validation:top5_accuracy', 'Regex': 'Top-5 Acc: ([0-9\\.]+)'},
                {'Name': 'train:loss', 'Regex': 'Train Loss: ([0-9\\.]+)'},
                {'Name': 'validation:loss', 'Regex': 'Val Loss: ([0-9\\.]+)'}
            ],
            use_spot_instances=False,  # Set to True for cost savings
            checkpoint_s3_uri=f's3://{self.bucket}/football-pass-prediction/checkpoints/{job_name}'
        )
        
        # Define training inputs
        train_input = TrainingInput(
            s3_data=s3_data_path,
            content_type='text/csv' if s3_data_path.endswith('.csv') else 'application/x-tar'
        )
        
        # Launch training job
        print("Starting training job...")
        estimator.fit({'train': train_input}, job_name=job_name)
        
        return estimator, job_name
    
    def deploy_model(self, estimator, endpoint_name=None, instance_type='ml.m5.large'):
        """Deploy trained model to SageMaker endpoint"""
        
        if endpoint_name is None:
            timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            endpoint_name = f'football-pass-prediction-endpoint-{timestamp}'
        
        print(f"Deploying model to endpoint: {endpoint_name}")
        print(f"Endpoint instance type: {instance_type}")
        
        # Deploy model
        predictor = estimator.deploy(
            initial_instance_count=1,
            instance_type=instance_type,
            endpoint_name=endpoint_name
        )
        
        print(f"Model deployed successfully to: {endpoint_name}")
        return predictor, endpoint_name
    
    def test_endpoint(self, predictor, test_data=None):
        """Test the deployed endpoint with sample data"""
        
        if test_data is None:
            # Create sample test data (you would replace this with real data)
            test_data = {
                'categorical_seq': [[[1, 2, 3, 4, 5, 6, 7] for _ in range(12)]],  # Sample categorical sequence
                'numerical_seq': [[[0.5, 0.3, 0.6, 0.4, 0.2, 0.1, 1.0, 1, 45.0, 0.0, 0.3, 1.0, 0.0] for _ in range(12)]],  # Sample numerical sequence
                'player_seq': [[[10, 15] for _ in range(24)]]  # Sample player sequence
            }
        
        print("Testing endpoint with sample data...")
        
        try:
            result = predictor.predict(test_data)
            print(f"Endpoint test successful!")
            print(f"Sample prediction: {result}")
            return result
        except Exception as e:
            print(f"Endpoint test failed: {e}")
            raise
    
    def cleanup_endpoint(self, predictor=None, endpoint_name=None):
        """Clean up SageMaker endpoint to avoid charges"""
        if predictor:
            predictor.delete_endpoint()
            print("Endpoint deleted successfully")
        elif endpoint_name:
            sagemaker_client = boto3.client('sagemaker', region_name=self.region)
            sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
            print(f"Endpoint {endpoint_name} deleted successfully")


def main():
    parser = argparse.ArgumentParser(description='Launch SageMaker training job for Advanced Football Pass Prediction')
    
    # Required arguments
    parser.add_argument('--data-path', type=str, required=True, 
                       help='Path to local training data (CSV file or directory)')
    
    # Optional arguments
    parser.add_argument('--job-name', type=str, help='Custom training job name')
    parser.add_argument('--instance-type', type=str, default='ml.p3.2xlarge',
                       help='SageMaker training instance type')
    parser.add_argument('--instance-count', type=int, default=1,
                       help='Number of training instances')
    parser.add_argument('--max-run-time', type=int, default=3600,
                       help='Maximum training time in seconds')
    
    # Model hyperparameters
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--window', type=int, default=12)
    parser.add_argument('--context-window', type=int, default=24)
    parser.add_argument('--d-model', type=int, default=256)
    parser.add_argument('--n-heads', type=int, default=8)
    parser.add_argument('--n-layers', type=int, default=6)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--csv-filename', type=str, default='data.csv')
    
    # Deployment options
    parser.add_argument('--deploy', action='store_true', 
                       help='Deploy model after training')
    parser.add_argument('--endpoint-instance-type', type=str, default='ml.m5.large',
                       help='Instance type for model endpoint')
    parser.add_argument('--test-endpoint', action='store_true',
                       help='Test endpoint after deployment')
    
    # AWS configuration
    parser.add_argument('--region', type=str, default='us-west-2',
                       help='AWS region for SageMaker')
    parser.add_argument('--role', type=str, 
                       help='SageMaker execution role ARN')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("AWS SageMaker Advanced Football Pass Prediction Job Launcher")
    print("=" * 80)
    
    try:
        # Initialize launcher
        launcher = SageMakerJobLauncher(region=args.region, role=args.role)
        
        # Upload data to S3
        s3_data_path = launcher.upload_data_to_s3(args.data_path)
        
        # Prepare hyperparameters
        hyperparameters = {
            'epochs': args.epochs,
            'batch-size': args.batch_size,
            'learning-rate': args.learning_rate,
            'weight-decay': args.weight_decay,
            'window': args.window,
            'context-window': args.context_window,
            'd-model': args.d_model,
            'n-heads': args.n_heads,
            'n-layers': args.n_layers,
            'dropout': args.dropout,
            'csv-filename': args.csv_filename
        }
        
        # Create and launch training job
        estimator, job_name = launcher.create_training_job(
            s3_data_path=s3_data_path,
            job_name=args.job_name,
            instance_type=args.instance_type,
            instance_count=args.instance_count,
            hyperparameters=hyperparameters,
            max_run_time=args.max_run_time
        )
        
        print(f"\nTraining job '{job_name}' completed successfully!")
        
        # Optional deployment
        if args.deploy:
            predictor, endpoint_name = launcher.deploy_model(
                estimator=estimator,
                instance_type=args.endpoint_instance_type
            )
            
            # Optional endpoint testing
            if args.test_endpoint:
                launcher.test_endpoint(predictor)
            
            print(f"\nEndpoint '{endpoint_name}' is ready for inference!")
            print(f"To delete the endpoint later, run:")
            print(f"aws sagemaker delete-endpoint --endpoint-name {endpoint_name}")
        
        print("\nJob launcher completed successfully!")
        
    except Exception as e:
        print(f"Job launcher failed: {e}")
        raise


if __name__ == '__main__':
    main()