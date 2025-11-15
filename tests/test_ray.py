"""
Usage:
    pytest tests/test_ray.py -s
"""

import os
import ray
import torch
import pytest

@pytest.fixture(scope="module")
def ray_init():
    # Set environment variables for testing
    os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,7"
    if not ray.is_initialized():
        ray.init(
            num_gpus=4,
            num_cpus=8,
            ignore_reinit_error=True,
            include_dashboard=False,
            log_to_driver=False
        )
    yield
    ray.shutdown()

def test_gpu_setup(ray_init):
    print("=== GPU Configuration Test ===")
    
    # Check CUDA_VISIBLE_DEVICES
    cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "Not set")
    print(f"CUDA_VISIBLE_DEVICES: {cuda_devices}")
    
    # Check PyTorch CUDA availability
    print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"PyTorch CUDA device count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
    
    print()

def test_ray_setup(ray_init):
    print("=== Ray Configuration Test ===")
    
    # Check Ray cluster resources
    resources = ray.cluster_resources()
    print(f"Ray cluster resources: {resources}")
    
    # Test a simple Ray remote function
    @ray.remote(num_gpus=1)
    def test_gpu_worker(worker_id):
        return {
            "worker_id": worker_id,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "Not set"),
            "available_gpus": ray.get_gpu_ids() if hasattr(ray, 'get_gpu_ids') else "Not available"
        }
    
    # Create 4 workers
    workers = [test_gpu_worker.remote(i) for i in range(4)]
    results = ray.get(workers)
    
    print("\nRay GPU worker results:")
    for result in results:
        print(f"  Worker {result['worker_id']}: {result}")
    
    print("Ray test completed successfully!")
    
    print()
