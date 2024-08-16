import datetime
def get_time():
	current_datetime = datetime.datetime.now()
	year = current_datetime.year
	month = current_datetime.month
	day = current_datetime.day
	hour = current_datetime.hour
	minute = current_datetime.minute
	second = current_datetime.second
	date_string = f"{year}-{month:02d}-{day:02d}_{hour:02d}-{minute:02d}-{second:02d}"
	return date_string

