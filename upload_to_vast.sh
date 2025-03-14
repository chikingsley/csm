#!/bin/bash
# Simplified script to upload the csm_upload directory to Vast.ai

# Set these variables
VAST_INSTANCE_ID="18825257"  # Your instance ID from Vast.ai

# Check if vastai CLI is installed
if ! command -v vastai &> /dev/null; then
  echo "Vast.ai CLI not found. Please install it with:"
  echo "pip install vast-ai-client"
  exit 1
fi

# Check if instance exists and get its status
echo "Checking instance status..."
INSTANCE_INFO=$(vastai show instances | grep $VAST_INSTANCE_ID)
if [ -z "$INSTANCE_INFO" ]; then
  echo "Instance $VAST_INSTANCE_ID not found. Please check your instance ID."
  exit 1
fi

INSTANCE_STATUS=$(echo $INSTANCE_INFO | awk '{print $3}')
echo "Current instance status: $INSTANCE_STATUS"

# If instance is not running (could be 'exited' or other states), ask to start it
if [ "$INSTANCE_STATUS" != "running" ]; then
  read -p "Instance is not running (status: $INSTANCE_STATUS). Would you like to start it? (y/n): " start_instance
  if [ "$start_instance" == "y" ] || [ "$start_instance" == "Y" ]; then
    echo "Starting instance $VAST_INSTANCE_ID..."
    vastai start instance $VAST_INSTANCE_ID
    echo "Waiting for instance to start (this may take a minute)..."
    sleep 45  # Give it more time to start
    
    # Check if it started successfully
    INSTANCE_INFO=$(vastai show instances | grep $VAST_INSTANCE_ID)
    INSTANCE_STATUS=$(echo $INSTANCE_INFO | awk '{print $3}')
    if [ "$INSTANCE_STATUS" != "running" ]; then
      echo "Instance failed to start (status: $INSTANCE_STATUS). Please check the Vast.ai console."
      exit 1
    fi
    echo "Instance started successfully!"
  else
    echo "Please start the instance manually before uploading files."
    exit 1
  fi
fi

# Get instance details again (in case it just started)
INSTANCE_INFO=$(vastai show instances | grep $VAST_INSTANCE_ID)
VAST_SSH_PORT=$(echo $INSTANCE_INFO | awk '{print $9}')
VAST_IP=$(echo $INSTANCE_INFO | awk '{print $8}')

echo "Instance details:"
echo "  IP: $VAST_IP"
echo "  SSH Port: $VAST_SSH_PORT"

# Ask for confirmation before uploading
echo ""
echo "Ready to upload files from ./csm_upload to Vast.ai instance."
read -p "Continue with upload? (y/n): " confirm_upload
if [ "$confirm_upload" != "y" ] && [ "$confirm_upload" != "Y" ]; then
  echo "Upload canceled."
  exit 0
fi

# Upload files
echo "Uploading files to Vast.ai instance..."
rsync -avz -e "ssh -p $VAST_SSH_PORT" csm_upload/ root@$VAST_IP:/workspace/csm/

echo ""
echo "Upload complete!"
echo "To connect to your instance:"
echo "  ssh -p $VAST_SSH_PORT root@$VAST_IP"
echo ""
echo "After connecting, run:"
echo "  cd /workspace/csm"
echo "  chmod +x setup_vast.sh  # If you included this file"
echo "  ./setup_vast.sh         # If you included this file"
echo ""
echo "Then access Jupyter at: http://$VAST_IP:8080"
