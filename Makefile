default:
	jupyter notebook

clean:
	python3 clean_data.py

clean2:
	python3 clean_data_2.py

cf-matrix:
	python3 build_cf_matrix.py

train:
	python3 model_training.py




profile-clean_data:
	python3 -m cProfile -o clean_data.prof clean_data.py

show-profile-clean_data:
	snakeviz clean_data.prof
