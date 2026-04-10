#!/bin/bash
echo "Deploying to Railway..."
railway up
echo "Deployment complete"
railway domain
