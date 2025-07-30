SHELL := /bin/bash
env: 
	python3 -m venv vps_env 
	source vps_env/bin/activate 
	pip3 install -e .

