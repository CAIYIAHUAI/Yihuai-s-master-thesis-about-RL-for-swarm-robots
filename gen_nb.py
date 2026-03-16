import json

cells = [
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "# Triangle Fill v4 Training Results\n",
        "This notebook parses `train.log` and visualizes the key training metrics."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "import pandas as pd\n",
        "import matplotlib.pyplot as plt\n",
        "import re\n",
        "\n",
        "# Parse the log file\n",
        "log_path = 'runs/triangle_fill_v4/train.log'\n",
        "\n",
        "data = []\n",
        "pattern = re.compile(r'update=\\s*(\\d+)\\s+mean_rew=([-\\d\\.]+).*?formation_loss=([-\\d\\.]+).*?formation_score=([-\\d\\.]+).*?collision_mean=([-\\d\\.]+).*?action_mean=([-\\d\\.]+).*?speed_mean=([-\\d\\.]+)')\n",
        "\n",
        "with open(log_path, 'r') as f:\n",
        "    for line in f:\n",
        "        m = pattern.search(line)\n",
        "        if m:\n",
        "            data.append({\n",
        "                'update': int(m.group(1)),\n",
        "                'mean_rew': float(m.group(2)),\n",
        "                'formation_loss': float(m.group(3)),\n",
        "                'formation_score': float(m.group(4)),\n",
        "                'collision_mean': float(m.group(5)),\n",
        "                'action_mean': float(m.group(6)),\n",
        "                'speed_mean': float(m.group(7))\n",
        "            })\n",
        "\n",
        "df = pd.DataFrame(data)\n",
        "\n",
        "if df.empty:\n",
        "    print('No data found in log. Make sure the path is correct.')\n",
        "else:\n",
        "    # Plotting\n",
        "    fig, axes = plt.subplots(3, 2, figsize=(15, 12))\n",
        "    fig.suptitle('Training Metrics from train.log (v4)', fontsize=16)\n",
        "\n",
        "    axes[0, 0].plot(df['update'], df['mean_rew'], label='Mean Reward', color='blue')\n",
        "    axes[0, 0].set_title('Mean Reward')\n",
        "    axes[0, 0].grid(True)\n",
        "    axes[0, 0].legend()\n",
        "\n",
        "    axes[0, 1].plot(df['update'], df['formation_loss'], label='Formation Loss', color='orange')\n",
        "    axes[0, 1].set_title('Formation Loss')\n",
        "    axes[0, 1].grid(True)\n",
        "    axes[0, 1].legend()\n",
        "\n",
        "    axes[1, 0].plot(df['update'], df['formation_score'], label='Formation Score', color='green')\n",
        "    axes[1, 0].set_title('Formation Score')\n",
        "    axes[1, 0].grid(True)\n",
        "    axes[1, 0].legend()\n",
        "\n",
        "    axes[1, 1].plot(df['update'], df['collision_mean'], label='Collision Mean', color='red')\n",
        "    axes[1, 1].set_title('Collision Mean')\n",
        "    axes[1, 1].grid(True)\n",
        "    axes[1, 1].legend()\n",
        "\n",
        "    axes[2, 0].plot(df['update'], df['action_mean'], label='Action Mean', color='purple')\n",
        "    axes[2, 0].set_title('Action Mean')\n",
        "    axes[2, 0].grid(True)\n",
        "    axes[2, 0].legend()\n",
        "\n",
        "    axes[2, 1].plot(df['update'], df['speed_mean'], label='Speed Mean', color='brown')\n",
        "    axes[2, 1].set_title('Speed Mean')\n",
        "    axes[2, 1].grid(True)\n",
        "    axes[2, 1].legend()\n",
        "\n",
        "    plt.tight_layout(rect=[0, 0.03, 1, 0.95])\n",
        "    plt.show()\n"
      ]
    }
  ]

notebook = {
  "cells": cells,
  "metadata": {
    "kernelspec": {
      "display_name": "yihuai",
      "language": "python",
      "name": "yihuai"
    },
    "language_info": {
      "name": "python"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 4
}

with open('/home/user/Yihuai/Code/yihuai-master-thesis/train_result.ipynb', 'w') as f:
    json.dump(notebook, f, indent=2)
