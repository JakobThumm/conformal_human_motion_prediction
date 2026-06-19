#!/bin/bash

# Script to run the Docker container for uncertainty quantification

set -e

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Export current user information for docker-compose
export USER_ID=$(id -u)
export GROUP_ID=$(id -g)
export USERNAME=$(whoami)

echo "Starting Docker container for conformal_human_motion_prediction..."
echo "User: $USERNAME (UID: $USER_ID, GID: $GROUP_ID)"

# Run using docker-compose
cd "$SCRIPT_DIR"
docker-compose up -d

echo ""
echo "Container started!"
echo ""
echo "First time? Inside the container run: bash docker/setup_env.sh (creates the 'unc' venv)."
echo ""
echo "To access the container shell, use:"
echo "  ./shell.sh   (or: docker exec -it chmp-dev bash)"
echo ""
echo "To stop the container:"
echo "  docker-compose down"