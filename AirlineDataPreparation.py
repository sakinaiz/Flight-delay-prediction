
# coding: utf-8

# # Load the libraries
import pandas as pd
import numpy as np
import datetime as dt

from sklearn.preprocessing import LabelBinarizer, LabelEncoder

#convert dates and times to datetime object
def convert_datetime(df, verbose = True):
	if verbose:
		print('\tConverting to datetime objects...')

	date_fmt = '%Y/%m/%d'
	time_fmt = '%H%M'

	date_of_flight = 'FL_DATE'
	departure_time = 'CRS_DEP_TIME'
	arrival_time = 'CRS_ARR_TIME'

	# prepend the times with 0 so it is in HHMM format
	df[departure_time] = ['{:04d}'.format(hhmm) for hhmm in df[departure_time]]
	df[arrival_time] = ['{:04d}'.format(hhmm) for hhmm in df[arrival_time]]
	# convert date and time to datetime objecs
	df[date_of_flight] = pd.to_datetime(df[date_of_flight], format= date_fmt)
	df[departure_time] = pd.to_datetime(df[departure_time], format= time_fmt)
	df[arrival_time]   = pd.to_datetime(df[arrival_time],   format= time_fmt)

	return df

# Remove columns which are not required 
def drop_junk_columns(df, valid_cols, verbose = True):
	if verbose:
		print('\tDeleting junk columns...')

	remove_col = list(set(df.columns) ^ set(valid_cols))
	df.drop(remove_col, axis=1, inplace=True);

	return df

# Remove cancelled and diverted flights
def drop_cancel_divert(df, verbose = True):
	remove_col = ['CANCELLED', 'DIVERTED']
	# change data types
	df['CANCELLED'] = df['CANCELLED'].astype('bool')
	df['DIVERTED']  =  df['DIVERTED'].astype('bool')

	if verbose:
		print('\tDeleting...\n\tCancelled flights = {0}\n\tDiverted flights = {1}'
	     	 .format(sum(df['CANCELLED']), sum(df['DIVERTED'])))
	
	select_index = (~df['CANCELLED']) & (~df['DIVERTED'])
	df = df.loc[select_index,:].copy()
	df.drop(remove_col, axis=1, inplace=True)

	return df

# Remove diverted flights
def drop_diverted(df, verbose = True):
	remove_col = ['DIVERTED']
	# change data types
	df['DIVERTED']  =  df['DIVERTED'].astype('bool')

	if verbose:
		print('\tDeleting Diverted flights = {}'
	    	  .format(sum(df['DIVERTED'])))

	df = df[~df['DIVERTED']].copy()
	df.drop(remove_col, axis=1, inplace=True)

	return df

# Modify delay times for cancelled flights
def modify_cancelled(df, verbose = True):
	# change data types
	df['CANCELLED'] = df['CANCELLED'].astype('bool')

	if verbose:
		print('\tModifying delays in Cancelled flights = {}'
	    	  .format(sum(df['CANCELLED'])))

	modify_index = df['CANCELLED']

	df.loc[modify_index, 'DEP_DELAY']       = 2880 # delay>2 days
	df.loc[modify_index, 'ARR_DELAY']       = 2880 # delay>2 days
	df.loc[modify_index, 'DEP_DELAY_GROUP'] = 13   # create new group
	df.loc[modify_index, 'ARR_DELAY_GROUP'] = 13   # create new group
	
	return df

# Filter out non-major airports from ORIGIN and DEST
def filter_airports(df, airports_df, verbose = True):
	if verbose:
		print('\tNumber of airports in original data = {}'
	    	  .format(len(set(df['ORIGIN']) | set(df['DEST']))))

	df = df.loc[df['ORIGIN'].isin(list(airports_df['Code'])), :]
	df =   df.loc[df['DEST'].isin(list(airports_df['Code'])), :]

	if verbose:
		print('\tNumber of airports in filtered data = {}'
	    	  .format(len(set(df['ORIGIN']) | set(df['DEST']))))

	# get regions of Origin airport
	df = df.merge(airports_df, left_on='ORIGIN', 
				  right_on='Code', how='left')
	df.rename(columns={'Region':'ORIG_REGION'}, inplace=True)
	df.drop(['Code', 'Airport'], axis=1, inplace=True)

	# get regions of Destination airport
	df = df.merge(airports_df, left_on='DEST', 
				  right_on='Code', how='left')
	df.rename(columns={'Region':'DEST_REGION'}, inplace=True)
	df.drop(['Code', 'Airport'], axis=1, inplace=True)

	return df

# Filter out non-major carriers from UNIQUE_CARRIER
def filter_airlines(df, carriers_df, verbose = True):
	if verbose:
		print('\tNumber of carriers in original data = {}'
	    	  .format(df['UNIQUE_CARRIER'].nunique()))

	df = df.loc[df['UNIQUE_CARRIER'].isin(list(carriers_df['Code'])), :]

	if verbose:
		print('\tNumber of carriers in filtered data = {}'
	    	  .format(df['UNIQUE_CARRIER'].nunique()))

	return df

# Check if any columns have null values
def check_nulls(df, verbose = True):
	if verbose:
		print('\tChecking for NULL values...')

	check_cols = ['FL_DATE', 'UNIQUE_CARRIER', 'FL_NUM',
	              'ORIGIN', 'DEST', 'DISTANCE',
	              'CRS_DEP_TIME', 'DEP_DELAY', 'DEP_DELAY_GROUP', 
	              'CRS_ARR_TIME', 'ARR_DELAY', 'ARR_DELAY_GROUP'
	             ]
	df_info_array = [df[check_cols].dtypes.values.T, 
					 df[check_cols].isnull().sum().values.T, 
	                 (100*df[check_cols].isnull().sum()/df.shape[0]).values.T]
	df_info = pd.DataFrame(df_info_array, 
	                       index=['column_type', 'null_count', '%_null_count'],
	                       columns = df[check_cols].columns)
	
	return df_info.loc['null_count'].sum()
	

# # Create target variable
def create_target(df, target, verbose = True):
	if verbose:
		print('\tCreating target columns...')

	# 1 indicates flight was delayed, 0 indicates flight was on time or early
	for thresh in np.arange(0,20,5):
		target_col_name = 'IS_DELAYED_ARR_' + str(thresh)
		df[target_col_name]  = df['ARR_DELAY']>thresh

	df['ARR_DELAY_GROUP'] = df['ARR_DELAY_GROUP'].astype('category')

	return df

def create_distance_bins(distance):
	if distance < 750.0:
		return 0
	elif distance < 1000.0:
		return 1
	else:
		return 2

def create_duration_bins(duration):
	if duration < 120.0:
		return 0
	elif duration < 180.0:
		return 1
	elif duration < 240.0:
		return 2
	else:
		return 3

def create_time_bins(timestamp):
	if timestamp.time() < dt.time(3,59):
		return "LateNight"
	elif timestamp.time() < dt.time(6,59):
		return "EarlyMorning"
	elif timestamp.time() < dt.time(11,59):
		return "Morning"
	elif timestamp.time() < dt.time(15,59):
		return "Afternoon"
	elif timestamp.time() < dt.time(22,59):
		return "Evening"
	else:
		return "LateNight"

 # function to get the time interval in miniutes between 2 timestamps
def get_inter_arrival_time(arr_time_tuple):
    if not any(arr_time_tuple.isnull()):
        current, previous = arr_time_tuple
        return (current - previous).seconds/60
    else:
        return 0

# # Create flight features
def create_flight_features(df, verbose=True):
	if verbose:
		print('\tCreating Flight features...')
	
	df["DISTANCE_BIN"] = df["DISTANCE"].apply(create_distance_bins).astype('category')
	df["DURATION_BIN"] = df["CRS_ELAPSED_TIME"].apply(create_duration_bins).astype('category')

	df['DEP_TIME_BIN'] = df['CRS_DEP_TIME'].apply(create_time_bins).astype('category')
	df['ARR_TIME_BIN'] = df['CRS_ARR_TIME'].apply(create_time_bins).astype('category')

	for thresh in np.arange(0,20,5):
		col_name = 'IS_DELAYED_DEP_' + str(thresh)
		df[col_name]  = df['DEP_DELAY']>thresh
	
	df['DEP_DELAY_GROUP'] = df['DEP_DELAY_GROUP'].astype('category')

	return df

def create_airport_feature(df, verbose = True):
	if verbose:
		print('\tCreating Airport features...')

	df_copy = df.groupby(['DEST','FL_DATE'],as_index=False).\
						 apply(lambda x: x.sort_values(["CRS_ARR_TIME"]))
	df_copy['PREV_CRS_ARR_TIME'] = df_copy['CRS_ARR_TIME'].shift(1)

	df_copy['INTER_ARR_TIME']     = df_copy[['CRS_ARR_TIME', 'PREV_CRS_ARR_TIME']]\
											.apply(get_inter_arrival_time, axis=1)
	df_copy['LOG_INTER_ARR_TIME'] = df_copy['INTER_ARR_TIME']\
                                  			.apply(lambda x : np.log(x) if x>0 else -5)

	df_copy.drop(['PREV_CRS_ARR_TIME'], axis=1,inplace=True)

	return df_copy

# # Create time based features
def create_time_features(df, holidays, verbose = True):
	if verbose:
		print('\tCreating Time-based features...')

	# create if date of flight is holiday
	df['HOLIDAY'] = df['FL_DATE'].isin(holidays)

	#Create day of week feature
	df['DAY_OF_WEEK'] = df['FL_DATE'].dt.weekday_name
	
	# ## Create a hour of day feature
	df['DEP_HOUR'] = df['CRS_DEP_TIME'].dt.hour.astype('category')
	df['ARR_HOUR'] = df['CRS_ARR_TIME'].dt.hour.astype('category')

	# ## Create a month feature
	df['MONTH'] = df['FL_DATE'].dt.month

	# ## Create a season feature
	df['SEASON'] = (df['MONTH']%12 + 3)//3
	seasons = {1:'Winter', 2:'Spring', 3:'Summer', 4:'Fall'}
	df['SEASON'] = df['SEASON'].map(seasons)

	# convert Month from number to string
	months = {1:'Jan',  2:'Feb',  3:'Mar',  4:'Apr',
    		  5:'May',  6:'Jun',  7:'Jul',  8:'Aug',
    		  9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'}
	df['MONTH'] = df['MONTH'].map(months)

	return df

# # Select features and target
def select_final_columns(df, features, target, extra, verbose = True):
	if verbose:
		print('\tCreating new dataframe...')

	new_df = df[sum([features, target, extra], [])]

	return new_df

# Change data types
def change_dtype(df, verbose = True):
	if verbose:
		print('\tChanging datatypes...')

	optimized_df = df.copy()

	df_int = df.select_dtypes(include=['int']).copy()
	#if verbose:
	#	print('\tInt:',list(df_int.columns.values))
	if list(df_int.columns.values):
	    df_int = df_int.apply(pd.to_numeric, downcast='unsigned')
	    optimized_df[df_int.columns] = df_int

	df_float = df.select_dtypes(include=['float']).copy()
	#if verbose:
	#	print('\tFloat:',list(df_float.columns.values))
	if list(df_float.columns.values):
	    df_float = df_float.apply(pd.to_numeric, downcast='float')
	    optimized_df[df_float.columns] = df_float

	df_obj = df.select_dtypes(include=['object']).copy()
	#if verbose:
	#	print('\tObject:',list(df_obj.columns.values))
	if list(df_obj.columns.values):
	    df_obj = df_obj.apply(pd.Series.astype, dtype='category')
	    optimized_df[df_obj.columns] = df_obj

	return optimized_df

def create_clean_df(df, airports_df, carriers_df, holidays, 
					features, target, extra, valid_cols, 
					verbose=True):
	#convert dates and times to datetime object
	df = convert_datetime(df, verbose)

	# Remove columns which are not required 
	df = drop_junk_columns(df, valid_cols, verbose)

	# drop cancelled and diverted flights
	df = drop_cancel_divert(df,verbose)

	# Drop the diverted flights
	#df = drop_diverted(df,verbose)

	# Modify cancelled flights
	#df = modify_cancelled(df, verbose)

	# # Filter out non-major airports from ORIGIN and DEST
	df = filter_airports(df, airports_df, verbose)

	# # Filter out non-major carriers from UNIQUE_CARRIER
	df = filter_airlines(df, carriers_df, verbose)

	# Check if any columns have null values
	num_null = check_nulls(df, verbose)
	if num_null != 0:
		print('\tThis month has {} null values'.format(num_null))

	# # Create target variable
	df = create_target(df, target, verbose=verbose)

	# Create flight features
	df = create_flight_features(df, verbose=verbose)

	# Create airport features
	df = create_airport_feature(df, verbose)

	# # Create time based features
	df = create_time_features(df, holidays, verbose)

	# # Select features and target
	new_df = select_final_columns(df, features, target, extra, verbose)

	# Change data types
	new_df = change_dtype(new_df, verbose)

	return new_df

# create dummy columns for categorical features
def create_category_as_dummy(df):
	category_cols = df.select_dtypes(include=['category']).columns

	df_dummy = df.select_dtypes(exclude=['category'])
	lb = LabelBinarizer()

	for col in category_cols:
		if df[col].nunique() > 2:
			tmp = pd.DataFrame( lb.fit_transform(df[col]), 
							columns=[col+'_'+c for c in lb.classes_],
							index = df.index )
			df_dummy = df_dummy.join (tmp)
		else:
			df_dummy[col] = (df[col] == df[col].unique()[0])

	return df_dummy


# create label columns for categorical features
def create_category_as_labels(df):
	category_cols = df.select_dtypes(include=['category']).columns

	df_copy = df.copy()
	le = LabelEncoder()
	for col in category_cols:
		df_copy.loc[:,col] = le.fit_transform(df[col])
		
	return df_copy

# create label columns for categorical features
def create_category_as_counts(df):
	category_cols = df.select_dtypes(include=['category']).columns

	df_copy = df.copy()
	for col in category_cols:
		counts  = df_copy[col].value_counts()
		counts  = counts.sort_index()
		counts  = counts.fillna(0)
		counts += np.random.rand(len(counts))/1000
		df_copy[col].cat.categories = counts

	return df_copy




