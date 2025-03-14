# Conversational Speech Model (CSM) Deployment Guide

This repository contains the implementation of the Conversational Speech Model (CSM), a state-of-the-art model for generating natural conversational speech based on the Llama-3.2-3B backbone.

## Table of Contents
- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Deployment on Vast.ai](#deployment-on-vastai)
  - [Finding a Suitable Instance](#finding-a-suitable-instance)
  - [Uploading Files](#uploading-files)
  - [Setting Up the Environment](#setting-up-the-environment)
  - [Training the Model](#training-the-model)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)

## Overview

The Conversational Speech Model (CSM) is designed to generate natural-sounding conversational speech that crosses the uncanny valley. It builds upon the Llama-3.2-3B foundation model and is fine-tuned for conversational voice generation.

## Prerequisites

- [Vast.ai](https://vast.ai/) account
- Vast.ai CLI installed (`pip install vast-ai-client`)
- SSH client
- Git

## Deployment on Vast.ai

### Finding a Suitable Instance

To find a suitable instance for training the CSM model, use the Vast.ai CLI with the following command:

```bash
vastai search offers 'reliability > 0.95 gpu_ram >= 16 dph < 1.5 num_gpus >= 1' -o 'dlperf_usd-'
```

This command searches for:
- Instances with reliability > 95%
- GPUs with at least 16GB RAM
- Cost less than $1.50 per hour
- At least 1 GPU
- Sorted by deep learning performance per dollar (best value first)

Recommended specifications:
- **GPU**: RTX 4080S, RTX 4090, RTX 5080, or similar
- **RAM**: At least 32GB
- **Storage**: 100GB+ for model weights and datasets
- **Cost**: $0.30-$0.80/hour for good performance/price ratio

### Uploading Files

1. **Prepare the upload directory**:
   The `csm_upload` directory should contain:
   - `generator.py`
   - `models.py`
   - `requirements.txt`
   - `train.py`
   - `model_weights/` (containing model files)

2. **Use the upload script**:
   The `upload_to_vast.sh` script automates the process of uploading files to your Vast.ai instance.

   ```bash
   # Make the script executable
   chmod +x upload_to_vast.sh
   
   # Run the script
   ./upload_to_vast.sh
   ```

   The script will:
   - Check if the Vast.ai CLI is installed
   - Verify your instance status
   - Start the instance if it's not running
   - Upload the files from the `csm_upload` directory

### Setting Up the Environment

Once the files are uploaded, SSH into your Vast.ai instance:

```bash
ssh -p <SSH_PORT> root@<SSH_IP>
```

Then run the setup script:

```bash
cd csm_upload
chmod +x setup_vast.sh
./setup_vast.sh
```

This will:
- Install required dependencies
- Set up the Python environment
- Prepare the model for training

### Training the Model

To start training the CSM model:

```bash
cd csm_upload
python train.py
```

You can monitor the training progress through the logs and use TensorBoard if configured.

## Project Structure

```
csm/
├── csm_upload/              # Main directory for upload
│   ├── generator.py         # Speech generation code
│   ├── models.py            # Model definitions
│   ├── requirements.txt     # Python dependencies
│   ├── train.py             # Training script
│   └── model_weights/       # Directory for model weights
├── upload_to_vast.sh        # Script to upload files to Vast.ai
├── setup_vast.sh            # Setup script for the Vast.ai instance
└── README.md                # This documentation
```

## Troubleshooting

### Common Issues

1. **Instance not found**:
   - Verify your instance ID in the `upload_to_vast.sh` script
   - Check if your instance is active in the Vast.ai console

2. **Upload failures**:
   - Ensure your instance is in the "running" state
   - Check your SSH connection details
   - Verify you have sufficient storage on the instance

3. **Training errors**:
   - Check CUDA availability with `torch.cuda.is_available()`
   - Ensure all dependencies are installed correctly
   - Verify model weights are properly loaded

### Vast.ai CLI Commands

Useful Vast.ai CLI commands:

```bash
# List your instances
vastai show instances

# Start an existing instance
vastai start instance <INSTANCE_ID>

# Stop an instance
vastai stop instance <INSTANCE_ID>

# Get SSH connection details
vastai show instances | grep <INSTANCE_ID>

# Create a new instance from an offer
vastai create instance <OFFER_ID> --image <DOCKER_IMAGE> --disk <DISK_SIZE_GB> --ssh --direct

# Example: Create a new instance with PyTorch
vastai create instance 18308720 --image vastai/pytorch:2.5.1-cuda-12.1.1 --disk 100 --ssh --direct

# Get SSH connection details to prove it was made
vastai show instances 
```

For more information, refer to the [Vast.ai documentation](https://vast.ai/docs/).
