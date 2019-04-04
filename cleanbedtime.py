import csv
import pandas as pd
import numpy as np
from dateutil.parser import parse

def time_parser(row):
	numbers = []
	comma = 0
	print(row)

	# make list containing all numbers of string
	for c in row:
		if c.isdigit():
			numbers.append(c)
		if c == ',':
			numbers.insert(0, '0')
			comma = 1
		print(numbers)

	# calculate number of numbers in list
	len_num = len(numbers)

	# when string doesn't contain any numbers return 'UNKNOWN'
	if len_num < 1:
		row = 'Unknown'
		return row

	# put numbers around entered numbers so it becomes the right format to parse
	if len_num == 1:
		numbers.insert(0, '0')
		numbers.append('0')
		numbers.append('0')

	if len_num == 2:
			numbers.append('0')
			numbers.append('0')

	if len_num == 3 and comma != 1:
		numbers.insert(0, '0')

	if len_num == 3 and comma == 1:
		numbers.append('0')

	if len_num > 1:
		if numbers[1] == ':':
			numbers.insert(0, '0')

	# if first two numbers are 24 or higher change to 00
	if len(numbers) == 4:
		if numbers[0] == '2' and numbers[1] == '4':
			numbers[0] = '0'
			numbers[1] = '0'
	
	# add ':' in the middle to get the right format for parsing
	numbers.insert(2, ':')
	row = ''.join(map(str, numbers[:5]))
	if numbers[0] == '1':
			if numbers[1] == '0' or numbers[1] == '1':
				row = row + ' pm'
	
	# parse all times to time format HH:MM:SS
	row = parse(row)
	row = row.time()
	print(row)
	return row

studentinfo = pd.read_csv('ODI-2019-csv.csv', sep=';')

bedtimes = (studentinfo['Time you went to be Yesterday'])

# put all bedtimes to MM:HH visualization assuming everyone went to bed after 15:00
for row in bedtimes:
	time_parser(row)
print("done")

# print(parse('2:30 am'))