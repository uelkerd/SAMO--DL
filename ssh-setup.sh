#!/bin/bash

# SSH Setup Script for GitHub Verified Commits
# This script sets up SSH authentication and commit signing for GitHub

echo "🚀 Setting up SSH for GitHub verified commits..."

# Start SSH agent if not running
if [ -z "$SSH_AGENT_PID" ]; then
    echo "📡 Starting SSH agent..."
    eval "$(ssh-agent -s)"
fi

# Export SSH agent environment variables for future sessions
echo "export SSH_AUTH_SOCK=$SSH_AUTH_SOCK" >> ~/.bashrc
echo "export SSH_AGENT_PID=$SSH_AGENT_PID" >> ~/.bashrc
echo "📝 Added SSH agent environment to ~/.bashrc"

# Add GitHub SSH key to agent
echo "🔑 Adding GitHub SSH key to agent..."
ssh-add ~/.ssh/id_github-0x_duelker

# Configure Git for SSH signing
echo "⚙️  Configuring Git for SSH commit signing..."
git config --global gpg.format ssh
git config --global user.signingkey ~/.ssh/id_github-0x_duelker.pub
git config --global commit.gpgsign true
git config --global tag.gpgsign true

# Create allowed signers file for signature verification
echo "📝 Creating allowed signers file..."
echo "156104354+uelkerd@users.noreply.github.com $(cat ~/.ssh/id_github-0x_duelker.pub)" > ~/.ssh/allowed_signers
git config --global gpg.ssh.allowedSignersFile ~/.ssh/allowed_signers

# Test connection
echo "🧪 Testing SSH connection to GitHub..."
ssh -T git@github.com

echo "✅ SSH setup complete!"
echo ""
echo "🔍 Next steps:"
echo "1. Make sure your SSH key is added to GitHub: https://github.com/settings/keys"
echo "2. Your commits will now be verified with your SSH key!"
echo "3. To start SSH agent in new terminals, run: eval \"\$(ssh-agent -s)\" && ssh-add ~/.ssh/id_github-0x_duelker"
echo "4. Run this script anytime: ./ssh-setup.sh"
