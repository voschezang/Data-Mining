import pandas as pd
from matplotlib import rcParams
import pickle
import numpy as np
import phonenumbers
from phonenumbers.phonenumberutil import region_code_for_country_code
import requests
import pycountry

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
    if output_as == 'boundingbox':
        lst = response[output_as]
        output = [float(i) for i in lst]
    if output_as == 'center':
        lst = [response.get(key) for key in ['lat','lon']]
        output = [float(i) for i in lst]
    return output

def country_id_to_country_longlat(country_id_numbers):
	visitor_country_id_name = []
	country_long_lat = []
	for id in country_id_numbers:
		print(id)
		id_region_code = region_code_for_country_code(id)
		
		# 'ZZ' denotes 'unknown or unspecified country'
		if id_region_code == 'ZZ':
			country_long_lat.append('')
			print("pass")
		else:
			visitor_country_id_name.append(id_region_code)
			country_info = pycountry.countries.get(alpha_2=id_region_code)
			if country_info != '':
				ll = get_boundingbox_country(country=country_info.name, output_as='center')
				country_long_lat.append(ll)

	return country_long_lat

data = pd.read_csv('data/training_set_VU_DM.csv', sep=',', nrows=1000)

country_id_numbers = data['visitor_location_country_id']

# visitor_ids = country_id_to_country_longlat(country_id_numbers)

visitor_country_id_name = []
country_long_lat = []
i = 0

for id in country_id_numbers:
	print(id)
	i +=1
	print(i, len(country_id_numbers))
	id_region_code = region_code_for_country_code(id)
	
	# 'ZZ' denotes 'unknown or unspecified country'
	if id_region_code == 'ZZ':
		country_long_lat.append('')
		print("pass")
	else:
		visitor_country_id_name.append(id_region_code)
		country_info = pycountry.countries.get(alpha_2=id_region_code)
		print(country_info.name)

		ll = get_boundingbox_country(country=country_info.name, output_as='center')
		country_long_lat.append(ll)

