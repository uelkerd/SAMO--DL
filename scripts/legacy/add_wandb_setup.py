#!/usr/bin/env python3
"""
Add WandB Setup
==============

This script adds proper wandb API key setup using Google Colab secrets
to avoid the manual API key prompt.
"""

import json

def add_wandb_setup():
    """Add wandb setup to the minimal notebook."""

    # Read the existing notebook
    with open('notebooks/MINIMAL_WORKING_TRAINING_COLAB.ipynb', 'r') as f:
        notebook = json.load(f)

    # Add wandb setup cell after the imports
    wandb_setup_cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 🔑 WANDB API KEY SETUP"
        ]
    }

    wandb_setup_code = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Setup Weights & Biases API key from Google Colab secrets\n",
            "import os\n",
            "import wandb\n",
            "\n",
            "print('🔑 SETTING UP WANDB API KEY')\n",
            "print('=' * 40)\n",
            "\n",
            "# Try to get API key from Colab secrets\n",
            "try:\n",
            "    from google.colab import userdata\n",
            "    \n",
            "    # Try different possible secret names\n",
            "    possible_secret_names = [\n",
            "        'WANDB_API_KEY',\n",
            "        'wandb_api_key',\n",
            "        'WANDB_KEY',\n",
            "        'wandb_key',\n",
            "        'WANDB_TOKEN',\n",
            "        'wandb_token'\n",
            "    ]\n",
            "    \n",
            "    api_key = None\n",
            "    used_secret_name = None\n",
            "    \n",
            "    for secret_name in possible_secret_names:\n",
            "        try:\n",
            "            api_key = userdata.get(secret_name)\n",
            "            used_secret_name = secret_name\n",
            "            print(f'✅ Found API key in secret: {secret_name}')\n",
            "            break\n",
            "        except:\n",
            "            continue\n",
            "    \n",
            "    if api_key:\n",
            "        # Set the environment variable\n",
            "        os.environ['WANDB_API_KEY'] = api_key\n",
            "        print(f'✅ API key set from secret: {used_secret_name}')\n",
            "        \n",
            "        # Test wandb login\n",
            "        try:\n",
            "            wandb.login(key=api_key)\n",
            "            print('✅ WandB login successful!')\n",
            "        except Exception as e:\n",
            "            print(f'⚠️ WandB login failed: {str(e)}')\n",
            "            print('Continuing without WandB...')\n",
            "    else:\n",
            "        print('❌ No WandB API key found in secrets')\n",
            "        print('\\n📋 TO SET UP WANDB SECRET:')\n",
            "        print('1. Go to Colab → Settings → Secrets')\n",
            "        print('2. Add a new secret with name: WANDB_API_KEY')\n",
            "        print('3. Value: Your WandB API key from https://wandb.ai/authorize')\n",
            "        print('4. Restart runtime and run this cell again')\n",
            "        print('\\n⚠️ Continuing without WandB logging...')\n",
            "        \n",
            "except ImportError:\n",
            "    print('⚠️ Google Colab secrets not available')\n",
            "    print('\\n📋 TO SET UP WANDB:')\n",
            "    print('1. Get your API key from: https://wandb.ai/authorize')\n",
            "    print('2. Run: wandb login')\n",
            "    print('3. Enter your API key when prompted')\n",
            "    print('\\n⚠️ Continuing without WandB logging...')\n",
            "\n",
            "print('\\n✅ WandB setup completed')"
        ]
    }

    # Find the imports cell and add wandb setup after it
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code' and 'import torch' in ''.join(cell['source']):
            # Insert wandb setup after imports
            notebook['cells'].insert(i + 2, wandb_setup_cell)
            notebook['cells'].insert(i + 3, wandb_setup_code)
            break

    # Also update the training arguments to disable wandb if no API key
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code' and 'TrainingArguments(' in ''.join(cell['source']):
            # Update training arguments to handle wandb properly
            cell['source'] = [
                "# Minimal training arguments - only essential parameters\n",
                "training_args = TrainingArguments(\n",
                "    output_dir='./minimal_emotion_model',\n",
                "    num_train_epochs=3,\n",
                "    per_device_train_batch_size=4,\n",
                "    per_device_eval_batch_size=4,\n",
                "    logging_steps=10,\n",
                "    save_steps=50,\n",
                "    eval_steps=50,\n",
                "    # Disable wandb if no API key is set\n",
                "    report_to=None if 'WANDB_API_KEY' not in os.environ else ['wandb']\n",
                ")\n",
                "\n",
                "print('✅ Minimal training arguments configured')\n",
                "if 'WANDB_API_KEY' in os.environ:\n",
                "    print('✅ WandB logging enabled')\n",
                "else:\n",
                "    print('⚠️ WandB logging disabled (no API key)')"
            ]
            break

    # Save the updated notebook
    with open('notebooks/MINIMAL_WORKING_TRAINING_COLAB.ipynb', 'w') as f:
        json.dump(notebook, f, indent=2)

    print('✅ Added WandB setup to minimal notebook!')
    print('📋 Changes made:')
    print('   ✅ Added WandB API key setup from Colab secrets')
    print('   ✅ Tries multiple possible secret names')
    print('   ✅ Graceful fallback if no API key found')
    print('   ✅ Updated training arguments to handle WandB properly')
    print('\\n📋 TO SET UP THE SECRET:')
    print('1. Go to Colab → Settings → Secrets')
    print('2. Add new secret:')
    print('   Name: WANDB_API_KEY')
    print('   Value: Your API key from https://wandb.ai/authorize')
    print('3. Restart runtime and run the notebook')

if __name__ == "__main__":
    add_wandb_setup()
