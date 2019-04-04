from dateutil.parser import parse

number_to_bedtime = {
    'one': '01:00:00',
    'two': '02:00:00',
    'three': '03:00:00',
    'four': '04:00:00',
    'five': '05:00:00',
    'six': '18:00:00',
    'seven': '19:00:00',
    'eight': '20:00:00',
    'nine': '21:00:00',
    'ten': '22:00:00',
    'eleven': '23:00:00',
    'twelve': '00:00:00'
}


def parse_bedtimes(fields=[]):
    return [parse_bedtime(field) for field in fields]


def parse_bedtime(field=''):
    # time in range 18:00 - 06:00
    # parse all times to time format HH:MM:SS
    time = parse_bedtime_helper(field)
    if time is not None:
        return parse(time, fuzzy=True).time()
    return None


def parse_bedtime_helper(field=''):
    # time in range 18:00 - 06:00
    row = field
    if row in number_to_bedtime.keys():
        return number_to_bedtime[row]

    numbers = []
    comma = 0

    # make list containing all numbers of string
    for c in row:
        if c.isdigit():
            numbers.append(c)
        if c == ',':
            numbers.insert(0, '0')
            comma = 1

    # calculate number of numbers in list
    len_num = len(numbers)

    # when string doesn't contain any numbers return 'UNKNOWN'
    if len_num < 1:
        return None

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

    return row
