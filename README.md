# BatCount-ResNet

Code for the paper: *Leveraging Pretrained ResNet Architectures for Enhanced Real-Time Bat Counting*.

## Setup

Clone the repository and install dependencies:

```bash
git clone https://github.com/bayazitkaraman/BatCount-ResNet.git
cd BatCount-ResNet

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate      # On Windows: .venv\Scripts\activate

# Install required packages
pip install -r requirements.txt

# Then run the main script:
python ResnetBatsCounting.py
