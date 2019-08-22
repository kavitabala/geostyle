import csv, sqlite3, numpy as np


class Annotations:
	def __init__(self,csv_name):
		self._create_db()
		data = np.genfromtxt(csv_name,delimiter=",",dtype=str)
		self._create_table(data[0])
		self._insert_data(data[1:])

	def _create_db(self):
		self.db_connection = sqlite3.connect(":memory:")
		self.db_cursor = self.db_connection.cursor()

	def _create_table(self,attributes):
		self.attributes = attributes
		self.total_attributes = attributes.shape[0]
		attribute_type = ['int','str','time','int','str','float','float','int','int','int','int','int','int','str','str','bool','bool','bool','str','str','str','bool','bool','bool','str']
		arg = ",".join([attributes[i]+" "+self._dtype(attribute_type[i]) for i in range(attributes.shape[0])])

		self.db_cursor.execute("CREATE TABLE StreetStyle (%s);"%arg)


	def get_attributes(self):
		data = self.db_cursor.execute('select * from StreetStyle')
		names = list(map(lambda x: x[0], data.description))
		return names

	def get_categories(self,attribute):
		data = self.db_cursor.execute('select DISTINCT '+attribute+' from StreetStyle')
		rows = data.fetchall()
		return [tmp[0] for tmp in rows if tmp[0]!='']

	def get_category_files(self,attribute,category):
		st = 'select url,x1,y1,x2,y2 from StreetStyle where '+attribute+' == "'+category+'"'
		# print(st)
		data = self.db_cursor.execute(st)
		rows = data.fetchall()
		return rows

	def get_all_files(self):
		st = 'select url,x1,y1,x2,y2 from StreetStyle'
		# print(st)
		data = self.db_cursor.execute(st)
		rows = data.fetchall()
		return rows

	def get_bounding_box(self,fname):
		st = 'select x1,y1,x2,y2 from StreetStyle where url LIKE '+'"%'+fname+'%"'
		# print(st)
		data = self.db_cursor.execute(st)
		rows = data.fetchall()
		return rows[0]


	def _insert_data(self,data):
		self.db_cursor.executemany("INSERT INTO StreetStyle VALUES (%s)"%",".join(['?']*self.total_attributes),data)

	def _dtype(self,dtype):
		if(dtype=='int'):
			return 'INT'
		if(dtype=='float'):
			return 'DOUBLE PRECISION'
		if(dtype=='str'):
			return 'VARCHAR(255)'
		if(dtype=='bool'):
			return 'VARCHAR(255)'
		if(dtype=='time'):
			return 'TIMESTAMP'

	def select(self,query):
		data = self.db_cursor.execute(query)
		rows = data.fetchall()
		return rows
