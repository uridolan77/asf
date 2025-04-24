#!/bin/bash
# Basic deployment script for staging environment

set -e

# Configuration
CONTAINER_REGISTRY="ghcr.io"
REPOSITORY_OWNER="your-org"  # Replace with your GitHub organization or username
IMAGE_NAME="asf-medical"
DEPLOYMENT_ENV=${1:-staging}  # Default to staging if not specified
SSH_USER="deploy"
SSH_HOST="your-staging-server.example.com"  # Replace with your server hostname

# Determine image tag based on environment
if [ "$DEPLOYMENT_ENV" = "production" ]; then
  TAG="latest"
else
  TAG=$(git rev-parse --short HEAD)
fi

echo "Deploying $IMAGE_NAME:$TAG to $DEPLOYMENT_ENV environment..."

# Pull the latest image
echo "Pulling latest image..."
ssh $SSH_USER@$SSH_HOST "docker pull $CONTAINER_REGISTRY/$REPOSITORY_OWNER/$IMAGE_NAME:$TAG"

# Create or update docker-compose.yml on the server
echo "Updating docker-compose configuration..."
cat > docker-compose.$DEPLOYMENT_ENV.yml << EOF
version: '3.8'

services:
  api:
    image: $CONTAINER_REGISTRY/$REPOSITORY_OWNER/$IMAGE_NAME:$TAG
    restart: always
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=\${DATABASE_URL}
      - REDIS_URL=\${REDIS_URL}
      - ENVIRONMENT=$DEPLOYMENT_ENV
      - LOG_LEVEL=INFO
    depends_on:
      - redis

  redis:
    image: redis:7
    restart: always
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes

volumes:
  redis_data:
EOF

scp docker-compose.$DEPLOYMENT_ENV.yml $SSH_USER@$SSH_HOST:~/docker-compose.yml
rm docker-compose.$DEPLOYMENT_ENV.yml

# Deploy using docker-compose
echo "Starting services..."
ssh $SSH_USER@$SSH_HOST "docker-compose down && docker-compose up -d"

echo "Deployment to $DEPLOYMENT_ENV completed successfully!"
