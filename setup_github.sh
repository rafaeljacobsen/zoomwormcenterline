#!/bin/bash

# GitHub Repository Setup Script
# This script helps you create a GitHub repository and push your code

echo "🚀 GitHub Repository Setup"
echo "=========================="

# Get repository name
read -p "Enter repository name (default: worm-segmentation): " REPO_NAME
REPO_NAME=${REPO_NAME:-worm-segmentation}

# Get GitHub username
read -p "Enter your GitHub username: " GITHUB_USERNAME

if [ -z "$GITHUB_USERNAME" ]; then
    echo "❌ GitHub username is required"
    exit 1
fi

echo ""
echo "Repository details:"
echo "  Name: $REPO_NAME"
echo "  Username: $GITHUB_USERNAME"
echo "  URL: https://github.com/$GITHUB_USERNAME/$REPO_NAME"
echo ""

# Check if SSH key exists
if [ ! -f ~/.ssh/id_rsa ] && [ ! -f ~/.ssh/id_ed25519 ]; then
    echo "⚠️  No SSH key found. You'll need to:"
    echo "1. Generate SSH key: ssh-keygen -t ed25519 -C 'your-email@example.com'"
    echo "2. Add to GitHub: https://github.com/settings/keys"
    echo "3. Test connection: ssh -T git@github.com"
    echo ""
    read -p "Have you set up SSH keys? (y/n): " SSH_READY
    if [ "$SSH_READY" != "y" ]; then
        echo "Please set up SSH keys first and run this script again."
        exit 1
    fi
fi

# Test SSH connection
echo "🔑 Testing SSH connection to GitHub..."
ssh -T git@github.com 2>&1 | grep -q "successfully authenticated"
if [ $? -eq 0 ]; then
    echo "✅ SSH connection successful"
else
    echo "❌ SSH connection failed. Please check your SSH setup."
    echo "Run: ssh -T git@github.com"
    exit 1
fi

# Configure git if not already done
git config --global user.name 2>/dev/null
if [ $? -ne 0 ]; then
    read -p "Enter your name for git commits: " GIT_NAME
    git config --global user.name "$GIT_NAME"
fi

git config --global user.email 2>/dev/null
if [ $? -ne 0 ]; then
    read -p "Enter your email for git commits: " GIT_EMAIL
    git config --global user.email "$GIT_EMAIL"
fi

# Set main as default branch
git branch -M main

# Add remote origin
echo "🔗 Adding GitHub remote..."
git remote add origin git@github.com:$GITHUB_USERNAME/$REPO_NAME.git

echo ""
echo "📋 Next steps:"
echo "1. Go to GitHub and create a new repository: https://github.com/new"
echo "2. Repository name: $REPO_NAME"
echo "3. Keep it public (or private if you prefer)"
echo "4. DON'T initialize with README, .gitignore, or license (we already have them)"
echo "5. After creating the repository, press Enter to push the code"
echo ""
read -p "Press Enter after creating the GitHub repository..."

# Push to GitHub
echo "🚀 Pushing code to GitHub..."
git push -u origin main

if [ $? -eq 0 ]; then
    echo "✅ Successfully pushed to GitHub!"
    echo ""
    echo "🎉 Your repository is now available at:"
    echo "   https://github.com/$GITHUB_USERNAME/$REPO_NAME"
    echo ""
    echo "📖 Share your repository:"
    echo "   git clone https://github.com/$GITHUB_USERNAME/$REPO_NAME.git"
else
    echo "❌ Failed to push to GitHub"
    echo "Make sure you've created the repository on GitHub first"
    exit 1
fi 