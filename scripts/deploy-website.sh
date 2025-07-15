#!/bin/bash

# Deploy website script - pushes changes and ensures GitHub Pages rebuild
# Usage: ./scripts/deploy-website.sh "commit message"

set -e  # Exit on error

# Color codes for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}🚀 Deploying website to GitHub Pages...${NC}"

# Check if commit message is provided
if [ -z "$1" ]; then
    echo -e "${RED}Error: Please provide a commit message${NC}"
    echo "Usage: $0 \"commit message\""
    exit 1
fi

# Add all changes
echo -e "${GREEN}📝 Adding changes...${NC}"
git add -A

# Commit with provided message
echo -e "${GREEN}💾 Committing changes...${NC}"
git commit -m "$1" || {
    echo -e "${YELLOW}No changes to commit${NC}"
    exit 0
}

# Push to GitHub
echo -e "${GREEN}⬆️  Pushing to GitHub...${NC}"
git push origin main

# Wait a moment for GitHub to process the push
sleep 2

# Trigger GitHub Pages rebuild
echo -e "${GREEN}🔨 Triggering GitHub Pages rebuild...${NC}"
gh api -X POST repos/dirvine/saorsa/pages/builds || {
    echo -e "${YELLOW}Warning: Could not trigger rebuild via API${NC}"
    echo -e "${YELLOW}GitHub Pages will rebuild automatically${NC}"
}

# Check build status
echo -e "${GREEN}📊 Checking build status...${NC}"
sleep 3
BUILD_STATUS=$(gh api repos/dirvine/saorsa/pages --jq '.status')
echo -e "Build status: ${YELLOW}${BUILD_STATUS}${NC}"

echo -e "${GREEN}✅ Deployment initiated successfully!${NC}"
echo -e "${GREEN}🌐 Your website will be updated at https://saorsalabs.com in a few minutes${NC}"

# Optional: Wait and check build completion
echo -e "\n${YELLOW}Waiting for build to complete...${NC}"
for i in {1..10}; do
    sleep 5
    STATUS=$(gh api repos/dirvine/saorsa/pages --jq '.status')
    if [ "$STATUS" = "built" ]; then
        echo -e "${GREEN}✅ Build completed successfully!${NC}"
        exit 0
    elif [ "$STATUS" = "errored" ]; then
        echo -e "${RED}❌ Build failed! Check GitHub Pages settings${NC}"
        exit 1
    fi
    echo -e "Build status: ${YELLOW}${STATUS}${NC} (attempt $i/10)"
done

echo -e "${YELLOW}Build still in progress. Check https://github.com/dirvine/saorsa/actions for status${NC}"