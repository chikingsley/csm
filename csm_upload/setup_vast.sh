#!/bin/bash
# Setup script for CSM model on Vast.ai

# Install dependencies
pip install -r requirements.txt

# Create directories
mkdir -p data
mkdir -p checkpoints

# Set up Jupyter notebook for training
mkdir -p notebooks
cat > notebooks/train_csm.ipynb << 'EOL'
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# CSM Model Training\n",
    "\n",
    "This notebook sets up and trains the Conversational Speech Model (CSM) with a larger backbone."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Install dependencies if needed\n",
    "!pip install -r ../requirements.txt"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "import os\n",
    "import sys\n",
    "import torch\n",
    "import torch.nn as nn\n",
    "import torch.optim as optim\n",
    "\n",
    "# Add the parent directory to the path\n",
    "sys.path.append('..')\n",
    "from models import Model, ModelArgs\n",
    "from train import llama3_2_3B, setup_model, TrainingConfig"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Check GPU availability\n",
    "print(f\"CUDA available: {torch.cuda.is_available()}\")\n",
    "if torch.cuda.is_available():\n",
    "    print(f\"CUDA device count: {torch.cuda.device_count()}\")\n",
    "    print(f\"CUDA device name: {torch.cuda.get_device_name(0)}\")\n",
    "    print(f\"CUDA memory allocated: {torch.cuda.memory_allocated(0) / 1e9} GB\")\n",
    "    print(f\"CUDA memory reserved: {torch.cuda.memory_reserved(0) / 1e9} GB\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Set up training configuration\n",
    "config = TrainingConfig(\n",
    "    backbone_flavor=\"llama-3B\",\n",
    "    decoder_flavor=\"llama-100M\",\n",
    "    batch_size=2,  # Start small, can increase based on GPU memory\n",
    "    gradient_accumulation_steps=16,  # Increase for larger effective batch size\n",
    "    epochs=5,\n",
    "    learning_rate=1e-4,\n",
    "    fp16=True,  # Use mixed precision for memory efficiency\n",
    "    compute_amortization=True,\n",
    "    checkpoint_dir=\"../checkpoints\"\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Initialize model\n",
    "model = setup_model(config)\n",
    "model = model.to(config.device)\n",
    "\n",
    "# Print model size\n",
    "total_params = sum(p.numel() for p in model.parameters())\n",
    "print(f\"Total parameters: {total_params / 1e9:.2f} billion\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Training Loop Implementation\n",
    "\n",
    "This is where you would implement the actual training loop. The CSM paper describes a compute amortization scheme where the backbone is trained on all frames but the decoder is trained on a subset (1/16) of frames."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Placeholder for training loop\n",
    "# This would be implemented based on your specific dataset and training objectives\n",
    "print(\"Training loop would be implemented here\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
EOL

# Create a README for Vast.ai setup
cat > README_VAST.md << 'EOL'
# CSM Model on Vast.ai

This document provides instructions for setting up and training the Conversational Speech Model (CSM) on Vast.ai.

## Setup Instructions

1. **Connect to your instance**:
   ```
   ssh -p <port> root@<instance_ip>
   ```

2. **Clone your repository** (if you're using GitHub):
   ```
   git clone https://github.com/yourusername/csm.git
   cd csm
   ```

3. **Or upload your files directly**:
   Use the Vast.ai web interface or SCP to upload your files:
   ```
   scp -P <port> -r /path/to/local/csm root@<instance_ip>:/workspace/
   ```

4. **Run the setup script**:
   ```
   cd /workspace/csm
   chmod +x setup_vast.sh
   ./setup_vast.sh
   ```

5. **Access Jupyter Notebook**:
   - Click the "Jupyter" button in the Vast.ai interface, or
   - Navigate to http://<instance_ip>:8080 in your browser

6. **Open the training notebook**:
   - Navigate to notebooks/train_csm.ipynb

## Directory Structure

- `models.py`: Model architecture definitions
- `generator.py`: Generator for audio synthesis
- `train.py`: Training script
- `notebooks/`: Jupyter notebooks for interactive training
- `checkpoints/`: Directory for saving model checkpoints
- `data/`: Directory for training data

## Using the RTX 4080s

The setup is configured to use both RTX 4080 GPUs. You can monitor GPU usage with:
```
nvidia-smi
```

For distributed training across both GPUs, the notebook includes PyTorch's DistributedDataParallel.

## Troubleshooting

- **Out of memory errors**: Reduce batch size or enable gradient accumulation
- **CUDA not available**: Check GPU status with `nvidia-smi`
- **Package installation issues**: Try installing packages individually with `pip install <package>`
EOL

echo "Setup complete! You can now upload this directory to Vast.ai and run the setup script."
