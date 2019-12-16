# A3C General Agent

## Setup (windows)

`pip install ./a3c_general_agent/requirements.txt`

### GPU Support

- Install CUDA 10.0
- Install cuDNN v7.6.5 for CUDA 10.0
- `pip install tensorflow-gpu==1.15` (might need to uninstall tensorflow first - unsure)

## Train
`python a3c_cartpole.py --train`

## Run
Train first and then:  
`python a3c_cartpole.py`
