import pandas as pd

studentinfo = pd.read_csv('ODI-2019-csv.csv', sep=';')
stress_levels = (studentinfo['What is your stress level (0-100)?'])

# cleans stress level values to integers between 0 and 100
def clean_stress_levels(value):
	# make list containing all numbers of string
	numbers = []
	for c in value:
		if c.isdigit():
			numbers.append(c)
		if c == ',':
			break
		if c == '-':
			numbers = [0]
			break
	
	# when there are no numbers inside the string assuma value = 50
	if numbers == []:
		numbers = [5, 0]
		value = int(''.join(map(str, numbers)))
	else:
		value = int(''.join(map(str, numbers)))

	# if value is over 100, put value to 100
	if value > 100:
		value = 100
	return value

# put each value in stress level data to 0-100
for value in stress_levels:
	
	# check if number is 'Nan'
	if type(value) is float:
		value = 100
	# if number is in string extract stress level
	elif type(value) is str:
		value = clean_stress_levels(value)



