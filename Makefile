default:
	jupyter notebook

clean:
	python3 clean_data.py > clean_data.txt
	cat clean_data.txt

clean-no-save:
	python3 clean_data.py

clean2a:
	python3 clean_data_2a_fit.py

clean2b:
	python3 clean_data_2b_transform.py

clean2c:
	python3 clean_data_2c_svd.py

clean2d:
	python3 clean_data_2d_testset.py

clean2e:
	python3 clean_data_2e_predict_testset.py

train:
	python3 model_training.py




profile-clean_data:
	python3 -m cProfile -o clean_data.prof clean_data.py

show-profile-clean_data:
	snakeviz clean_data.prof
