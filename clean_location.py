import pandas as pd
from matplotlib import rcParams
import pickle
import numpy as np
import phonenumbers
from phonenumbers.phonenumberutil import region_code_for_country_code
import requests
import pycountry
import math

np.random.seed(123)

rcParams['font.family'] = 'serif'
rcParams['font.size'] = 14

# https://gis.stackexchange.com/questions/212796/get-lat-lon-extent-of-country-from-name-using-python
def get_boundingbox_country(country, output_as='boundingbox'):
    """
    get the bounding box of a country in EPSG4326 given a country name

    Parameters
    ----------
    country : str
        name of the country in english and lowercase
    output_as : 'str
        chose from 'boundingbox' or 'center'. 
         - 'boundingbox' for [latmin, latmax, lonmin, lonmax]
         - 'center' for [latcenter, loncenter]

    Returns
    -------
    output : list
        list with coordinates as str
    """
    # create url
    url = '{0}{1}{2}'.format('http://nominatim.openstreetmap.org/search?country=',
                             country,
                             '&format=json&polygon=0')
    response = requests.get(url).json()[0]

    # parse response to list
    # if output_as == 'boundingbox':
    #     lst = response[output_as]
    #     output = [float(i) for i in lst]
    if output_as == 'center':
        lst = [response.get(key) for key in ['lat','lon']]
        output = [float(i) for i in lst]
    return output

data = pd.read_csv('data/training_set_VU_DM.csv', sep=',', nrows=1000)

country_id_numbers = data['visitor_location_country_id']

countries_long_lat = {} 
countries_long_lat[-1] = ''

for id in country_id_numbers.unique():
	id_region_code = region_code_for_country_code(id)
	
	# 'ZZ' denotes 'unknown or unspecified country'
	if id_region_code == 'ZZ':
		# countries_long_lat['ZZ'] = ''
		pass
	else:
		country_info = pycountry.countries.get(alpha_2=id_region_code)

		# get longitudal and latitudal coordinates of country
		ll = get_boundingbox_country(country=country_info.name, output_as='center')
		
		# key is the country id number
		countries_long_lat[id] = ll

with open('countries_long_lat.pkl', 'wb') as pickle_file:
	pickle.dump(countries_long_lat, pickle_file)

def calculate_distance(a, b):
	'''
	Calculate the distance from point a to point b.
	Variables a and b are tuples containing a longitudal and a latitudal coördinate.
	'''

	# approximate radius of earth in km
	R = 6373.0

	lata = math.radians(a[0])
	lona = math.radians(a[1])
	latb = math.radians(b[0])
	lonb = math.radians(b[1])

	distance_lon = lonb - lona
	distance_lat = latb - lata

	afs = math.sin(distance_lat / 2)**2 + math.cos(lata) * math.cos(latb) * math.sin(distance_lon / 2)**2
	cir = 2 * math.atan2(math.sqrt(afs), math.sqrt(1 - afs))

	distance = R * cir

	# print("Result:", distance)
	return distance

def make_distance_matrix(countries_long_lat):
	'''
	Makes a distance matrix of all countries existing in the dictionary. 
	Puts all keys with unknown 
	'''
	n_o_countries = len(countries_long_lat.keys())+1
	key_number = 1
	distance_matrix = np.zeros((n_o_countries, n_o_countries))

	# assign rows and columns to keys
	for key in countries_long_lat:
		distance_matrix[key_number][0] = key
		distance_matrix[0][key_number] = key
		key_number += 1

	i = 1
	j = 1
	for key1 in countries_long_lat:
		i = 1
		for key2 in countries_long_lat:
			# if one of the keys is -1 no distance data is available
			if key1 == -1 or key2 == -1:
				distance_matrix[i][j] = np.nan
				i += 1

			# if the countries pointed by the key are the samen distance is 0
			elif key1 == key2:
				distance_matrix[i][j] = 0
				i += 1

			# else calculate distance from key1 to key2 and put in matrix  
			else:
				c1 = countries_long_lat[key1]
				c2 = countries_long_lat[key2]
				distance_matrix[i][j] = calculate_distance(c1, c2)
				distance_matrix[j][i] = calculate_distance(c1, c2)
				i += 1
		j+=1

	# print(distance_matrix)
	return distance_matrix

distance_matrix = make_distance_matrix(countries_long_lat)
print(distance_matrix)

# with open('countries_long_lat.pkl', 'rb') as pickle_file:
# 	new_data = pickle.load(pickle_file)