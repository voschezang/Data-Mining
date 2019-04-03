import csv
import pandas as pd
import numpy as np
from dateutil.parser import parse

def time_parser(row):
	numbers = []
	print(row)

	for c in row:
		if c.isdigit():
			numbers.append(c)
		if c == ',':
			numbers.insert(0, '0')

	if len(numbers) < 1:
		row = '00:00'
		row = parse(row)
		row = row.time()
		return row

	if len(numbers) == 1:
		numbers.insert(0, '0')
		numbers.append('0')
		numbers.append('0')

	if len(numbers) == 2:
			numbers.append('0')
			numbers.append('0')

	if len(numbers) == 3:
		numbers.insert(0, '0')

	if len(numbers) > 1:
		if numbers[1] == ':':
			numbers.insert(0, '0')

	if len(numbers) == 4:
		if numbers[0] == '2' and numbers[1] == '4':
			numbers[0] = '0'
			numbers[1] = '4'
	
	numbers.insert(2, ':')
	row = ''.join(map(str, numbers[:5]))
	if numbers[0] == '1':
			if numbers[1] == '0' or numbers[1] == '1':
				row = row + ' pm'

	row = parse(row)
	row = row.time()
	print(row)
	return row


	# if row == '110305pm' or row == 'x':
	# 	print("fukyou")
	# 	row = '00:00'
	# if row == '23u' or row == '23:00 uur':
	# 	row = '23:00'
	# if '~' in row:
	# 	row = row.replace('~', '')
	# 	print(row)
	
	# if 'am' in row:
	# 	row = row.replace('am', '')
	# 	print(row)

	# if len(row) == 1:
	# 	if int(row[0]) <= 6:
	# 		row = '0' + row[0] + " am"
	# 	if int(row[0]) > 6:
	# 		row = '0' + row + " pm"

	# if ',' or '.' in row:
	# 	row = row.replace(',', ':')
	# 	row = row.replace('.', ':')
	# 	if len(row) == 4 and row[2] == ':':
	# 		row = row + '0'
	# 	if 'p:m' in row:
	# 		print("ko")
	# 		row = row.replace('p:m', 'pm')
	# 	if 'a:m' in row:
	# 		row = row.replace('a:m', 'am')

	# if row.isdigit():
	# 	if len(row) > 3:
	# 		row = row[:2] + ':' + row[2:]
	# 	elif len(row) == 3 and row[0] > 2:
	# 		row = '0' + row[:1] + ':' + row[2:]
	# 	else:
	# 		row = row[:2] + ':00'

	# if ':' in row and len(row) == 5:
	# 	if int(row[1]) >= 4 and int(row[0]) >= 2:
	# 		row = '00'+ row[:2]

	# if not any(c.isdigit() for c in row):
	# 	print("aj")
	# 	row = '00:00'

studentinfo = pd.read_csv('ODI-2019-csv.csv', sep=';')

bedtimes = (studentinfo['Time you went to be Yesterday'])

# put all bedtimes to MM:HH visualization assuming everyone went to bed after 15:00
for row in bedtimes:
	time_parser(row)
print("done")

# print(parse('2:30 am'))