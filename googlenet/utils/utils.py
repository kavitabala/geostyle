from os.path import join
from urllib.parse import urlparse
def get_filename_from_url(url,directory):
	fname = urlparse(url).path.split("/")[-1]
	subdirs = [fname[tmp] for tmp in range(3)]
	if(directory==""):
		dfname = join(subdirs[0],subdirs[1],subdirs[2],fname)
	else:
		dfname = join(directory,subdirs[0],subdirs[1],subdirs[2],fname)
	return dfname