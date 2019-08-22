import logging

class Logger:
	def __init__(self,name):
		self.logger = logging.getLogger(name)
		self.logger.setLevel(logging.DEBUG)
		fh = logging.FileHandler(name+".log")
		fh.setLevel(logging.INFO)
		ch = logging.StreamHandler()
		ch.setLevel(logging.INFO)
		formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
		fh.setFormatter(formatter)
		ch.setFormatter(formatter)
		
		self.logger.addHandler(fh)
		self.logger.addHandler(ch)
