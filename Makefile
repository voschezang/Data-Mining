default:
	jupyter notebook

clean:
	python3 clean_data.py

train:
	python3 model_training.py
