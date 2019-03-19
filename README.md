# DDPG with Curiosity Driven Exploration and Multi-Criteria Hindsight Experience Replay

(modified from OpenAI Baselines commit #[3900f2a4473ce6b26a8129372ca8d5e02c766c9c](https://github.com/openai/baselines/tree/3900f2a4473ce6b26a8129372ca8d5e02c766c9c))

## Prerequisites 
Baselines requires python3 (>=3.5) with the development headers. You'll also need system packages CMake, OpenMPI and zlib. Those can be installed as follows
### Ubuntu
```bash
sudo apt-get update && sudo apt-get install cmake libopenmpi-dev python3-dev zlib1g-dev
```
    
### Mac OS X
Installation of system packages on Mac requires [Homebrew](https://brew.sh). With Homebrew installed, run the follwing:
```bash
brew install cmake openmpi
```
    
## Virtual environment
From the general python package sanity perspective, it is a good idea to use virtual environments (virtualenvs) to make sure packages from different projects do not interfere with each other. You can install virtualenv (which is itself a pip package) via
```bash
pip install virtualenv
```
Virtualenvs are essentially folders that have copies of python executable and all python packages.
To create a virtualenv called venv with python3, one runs 
```bash
virtualenv /path/to/venv --python=python3
```
To activate a virtualenv: 
```
. /path/to/venv/bin/activate
```
More thorough tutorial on virtualenvs and options can be found [here](https://virtualenv.pypa.io/en/stable/) 


## Installation
Clone the repo and cd into it:
```bash
git clone https://github.com/CDMCH/ddpg-with-curiosity-and-multi-criteria-her.git
cd ddpg-with-curiosity-and-multi-criteria-her
```
If using virtualenv, create a new virtualenv and activate it
```bash
virtualenv env --python=python3
. env/bin/activate
```
Install baselines package
```bash
pip install -e .
```
### Block Stacking Environments
The block stacking environments associated with the paper can be found [here](https://github.com/CDMCH/gym-fetch-stack).

They use the [MuJoCo](http://www.mujoco.org) physics simulator, which is proprietary and requires binaries and a license (temporary 30-day license can be obtained from [www.mujoco.org](http://www.mujoco.org)). Instructions on setting up MuJoCo can be found [here](https://github.com/openai/mujoco-py)

## Example training script

After installing the block stacking environments, you can run the example script to train an agent to stack 2 blocks with sparse rewards:
```
./train_on_stack2_sparse_full_curriculum_curiosity_multi_criteria.sh
```
Or visualize the pretrained agents with:
```
./watch_stack2.sh
./watch_stack3.sh
./watch_stack4.sh
```
