import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import linregress

from numpy import sin,exp
from key44 import cities as cname,subcontinent
import matplotlib.pyplot as plt

from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.vector_ar.var_model import VAR
from sklearn.neural_network import MLPRegressor


################################################################################
import pickle
#Pickle functions
def unpickle(file):
	with open(file, 'rb') as fo:
		dict = pickle.load(fo)
	return dict

def inpickle(diction,file):
	with open(file, 'wb') as fo:
		pickle.dump(diction,fo)


################################################################################

keys = unpickle('key.pkl')
attributes = keys['attributes']
categories = keys['categories']

################################################################################
sinbounds = ([0, 1/72, -3.14, 0], [1, 1/6, 3.14, 1])

sinlbounds = ([0,0, 1/72, -3.14, 0,-1/100,0], [1,1, 1/6, 3.14, 1,1/100,1])

cycusbounds = ([0, 0, 1/72, -3.14, 0], [1, 20, 1/6, 3.14, 1])
cycdsbounds = ([-1, 0, 1/72, -3.14, -1], [0, 20, 1/6, 3.14, 0])


usbounds = ([0.5, 0, 0, 1/72, -3.14, 0, -1/100, 0], [1, 1, 20, 1/6, 3.14, 1, 1/100, 1])
dsbounds = ([0, -1, 0, 1/72, -3.14, -1, -1/100, 0], [1, 0, 20, 1/6, 3.14, 1, 1/100, 1])
################################################################################


def linear(x,m,c):
	return m*x+c


def sinusoidal(x,m1,f,o,b1):
	return (m1*(sin(f*x+o))+b1)


def sinusoidal_lin(x,r,m1,f,o,b1,m2,b2):
	return r*(m1*(sin(f*x+o))+b1)+(1-r)*(m2*x+b2)


def cyclic(x,m1,w,f,o,b1):
	return (m1*(exp(w*sin(f*x+o))/exp(w)+b1))


def spike(x,r,m1,w,f,o,b1,m2,b2):
	return (r*m1*(exp(w*sin(f*x+o))/exp(w)+b1) + (1-r)*(m2*x+b2))
################################################################################


def mean_pred(data,gap=0,predtill=1):
	assert predtill-1<=gap
	true = data[:,-predtill:,:]
	pred = np.repeat(np.mean(data[:,:-gap-1,:],axis=1,keepdims=True),predtill,axis=1)
	mae = np.mean(np.abs(pred-true))
	mape = np.mean(np.abs(pred-true)/true)*100
	return mae, mape,pred

def last_pred(data,gap=0,predtill=1):
	assert predtill-1<=gap
	true = data[:,-predtill:,:]
	# pred = data[:,-2,:]
	pred = np.repeat(np.expand_dims(data[:,-2-gap,:],axis=1),predtill,axis=1)
	mae = np.mean(np.abs(pred-true))
	mape = np.mean(np.abs(pred-true)/true)*100
	return mae, mape,pred

#TODO implement gap
def exp_smooth(data,alpha):
	true = data[:,-1,:]
	pred = data[:,:-1,:]
	alphas = np.ones((1,pred.shape[1],1))*alpha
	for i in range(2,pred.shape[1]+1):
		alphas[0,-i,0] = (1-alpha)*alphas[0,-i+1,0]
	alphas[0,0,0] = alphas[0,0,0]/alpha

	pred = np.sum(pred*alphas,axis=1)

	mae = np.mean(np.abs(pred-true))
	mape = np.mean(np.abs(pred-true)/true)*100
	return mae, mape, pred

#TODO implement gap
def exp_smooth_final(data,gap=0,predtill=1):
	assert predtill-1<=gap
	malpha = 0
	mmae = 100
	for i in range(100):
		alpha = i*0.01+0.01
		mae,_,_ = exp_smooth(data[:,:-1-gap,:],alpha)
		if(mae<mmae):
			mmae = mae
			malpha = alpha
	if(gap!=0):
		tmp = data[:,:-gap,:]
	else:
		tmp = data[:,:,:]
	for i in range(gap+1):
		_,_,pred = exp_smooth(tmp,malpha)
		tmp = np.concatenate([tmp[:,:-1,:],np.expand_dims(pred,axis=0),np.expand_dims(data[:,i-gap,:],axis=1)],axis=1)
		# tmp = np.concatenate([tmp[:,:-1,:],np.expand_dims(pred,axis=0)],axis=1)
	true = data[:,-predtill:,:]
	pred = tmp[:,-predtill-1:-1,:]
	mae = np.mean(np.abs(pred-true))
	mape = np.mean(np.abs(pred-true)/true)*100
	return mae, mape, pred

def ar(data,gap=0,predtill=1):
	assert predtill-1<=gap
	true = data[:,-predtill:,:]
	pred = []
	for i in range(data.shape[2]):
		arm = AR(data[0,:-gap-1,i])
		fitted = arm.fit()
		# print("Lag", fitted.k_ar)
		# print("Coefficients", fitted.params)
		pred.append(fitted.predict(start=data.shape[1]-gap-1,end=data.shape[1]-1)[gap-predtill:gap])
	pred = np.expand_dims(np.array(pred).T,axis=0)
	mae = np.mean(np.abs(pred-true))
	mape = np.mean(np.abs(pred-true)/true)*100
	return mae, mape, pred

#buggy
def arima(data,gap=0,predtill=1):
	assert predtill-1<=gap
	true = data[:,-predtill:,:]
	pred = []
	for i in range(data.shape[2]):
		arimam = ARIMA(data[0,:-1,i],order=(7,0,3))
		fitted = arimam.fit()
		# print("Lag", fitted.k_ar)
		# print("Coefficients", fitted.params)
		pred.append(fitted.predict(start=data.shape[1]-gap-1,end=data.shape[1]-1)[gap-predtill:gap])
	pred = np.expand_dims(np.array(pred).T,axis=0)
	mae = np.mean(np.abs(pred-true))
	mape = np.mean(np.abs(pred-true)/true)*100
	return mae, mape, pred
#buggy
def var(data,gap=0,predtill=1):
	assert predtill-1<=gap
	true = data[:,-predtill:,:]
	pred = []
	# varm = VAR(data[0,:-1,:])
	varm = VAR(data[0,:-gap-1,:])
	fitted = varm.fit()
	# print("Lag", fitted.k_ar)
	print("Coefficients", fitted.params.shape)
	pred = varm.predict(fitted.params,start=data.shape[1]-gap-2,end=data.shape[1]-1)[gap+1-predtill:gap+1]
	print(pred.shape)
	pred = np.expand_dims(np.array(pred),axis=0)
	mae = np.mean(np.abs(pred-true))
	mape = np.mean(np.abs(pred-true)/true)*100
	return mae, mape, pred


def fit_curve_lin(data,confs,gap=0,predtill=1):
	assert predtill-1<=gap
	true = data[:,-predtill:,:]
	pred = []

	for i in range(data.shape[2]):
		trend = data[0,:-1-gap,i]
		conf = confs[0,:-1-gap,i]

		errs = []
		errf = 5000
		fitf = None
		fittrend = None
		fits = []

		weeks = list(range(trend.shape[0]))
		try:
			params,_ = curve_fit(linear,range(len(weeks)),trend,method='lm',p0=(0,0),sigma=conf)
			# s,i,_,_,_ = linregress(range(len(weeks)),linear,trend,method='lm',p0=(0,0),bounds=([-1/100,0],[1/100,1]),sigma=conf)
			err_l = np.sqrt(np.sum(np.square(np.array([linear(tmp,params[0],params[1]) for tmp in range(len(weeks))])-trend)))
			errs.append(err_l)
			fits.append(params)
			if(err_l<errf):
				fit = "linear"
				errf = err_l
				fitf = params
				fittrend = [linear(tmp,params[0],params[1]) for tmp in range(len(weeks))]
				fittrend_l = [linear(tmp,params[0],params[1]) for tmp in range(len(weeks))]
				pred_tmp = [linear(tmp,params[0],params[1]) for tmp in range(data.shape[1]-predtill,data.shape[1])]
		except:
			raise
		if(i%100==0):
			print("Done",i,"/",data.shape[2])
		pred.append(pred_tmp)
	pred = np.expand_dims(np.array(pred).T,axis=0)
	mae = np.mean(np.abs(pred-true))
	mape = np.mean(np.abs(pred-true)/true)*100
	return mae, mape, pred

def fit_curve_sin(data,confs,gap=0,predtill=1):
	assert predtill-1<=gap
	true = data[:,-predtill:,:]
	pred = []

	for i in range(data.shape[2]):
		trend = data[0,:-1-gap,i]
		conf = confs[0,:-1-gap,i]

		errs = []
		errf = 5000
		fitf = None
		fittrend = None
		fits = []

		weeks = list(range(trend.shape[0]))
		try:
			params,_ = curve_fit(linear,range(len(weeks)),trend,method='lm',p0=(0,0),sigma=conf)
			# s,i,_,_,_ = linregress(range(len(weeks)),linear,trend,method='lm',p0=(0,0),bounds=([-1/100,0],[1/100,1]),sigma=conf)
			err_l = np.sqrt(np.sum(np.square(np.array([linear(tmp,params[0],params[1]) for tmp in range(len(weeks))])-trend)))
			errs.append(err_l)
			fits.append(params)
			if(err_l<errf):
				fit = "linear"
				errf = err_l
				fitf = params
				fittrend = [linear(tmp,params[0],params[1]) for tmp in range(len(weeks))]
				fittrend_l = [linear(tmp,params[0],params[1]) for tmp in range(len(weeks))]
				pred_tmp = [linear(tmp,params[0],params[1]) for tmp in range(data.shape[1]-predtill,data.shape[1])]
		except:
			raise
		try:
			params,_ = curve_fit(sinusoidal,range(len(weeks)),trend,method='trf',p0=(np.max(trend)-np.min(trend),1/8,0,(np.max(trend)+np.min(trend))/2),bounds=sinbounds,sigma=conf)
			m1,f,o,b1 = params[0],params[1],params[2],params[3]
			err1 = np.sqrt(np.sum(np.square(np.array([sinusoidal(tmp,m1,f,o,b1) for tmp in range(len(weeks))])-trend)))
			errs.append(err1)
			fits.append(params)
			# print(err1<errf and err1<0.9*err_l,err1<0.9*err_l,errf,err1,0.9*err_l)
			if(err1<errf and err1<explain_factor*err_l):
				fit = "upspike"
				fitf = params
				fittrend_sin = [sinusoidal(tmp,m1,f,o,b1) for tmp in range(len(weeks))]
				pred_tmp = [sinusoidal(tmp,m1,f,o,b1) for tmp in range(data.shape[1]-predtill,data.shape[1])]
			else:
				fittrend_sin = fittrend_l
		except:
			fittrend_sin = fittrend_l
			pass

		if(i%100==0):
			print("Done",i,"/",data.shape[2])
		pred.append(pred_tmp)
	pred = np.expand_dims(np.array(pred).T,axis=0)
	mae = np.mean(np.abs(pred-true))
	mape = np.mean(np.abs(pred-true)/true)*100
	return mae, mape, pred

def fit_curve_sinl(data,confs,gap=0,predtill=1):
	assert predtill-1<=gap
	true = data[:,-predtill:,:]
	pred = []

	for i in range(data.shape[2]):
		trend = data[0,:-1-gap,i]
		conf = confs[0,:-1-gap,i]

		errs = []
		errf = 5000
		fitf = None
		fittrend = None
		fits = []

		weeks = list(range(trend.shape[0]))
		try:
			params,_ = curve_fit(linear,range(len(weeks)),trend,method='lm',p0=(0,0),sigma=conf)
			# s,i,_,_,_ = linregress(range(len(weeks)),linear,trend,method='lm',p0=(0,0),bounds=([-1/100,0],[1/100,1]),sigma=conf)
			err_l = np.sqrt(np.sum(np.square(np.array([linear(tmp,params[0],params[1]) for tmp in range(len(weeks))])-trend)))
			errs.append(err_l)
			fits.append(params)
			if(err_l<errf):
				fit = "linear"
				errf = err_l
				fitf = params
				fittrend = [linear(tmp,params[0],params[1]) for tmp in range(len(weeks))]
				fittrend_l = [linear(tmp,params[0],params[1]) for tmp in range(len(weeks))]
				pred_tmp = [linear(tmp,params[0],params[1]) for tmp in range(data.shape[1]-predtill,data.shape[1])]
		except:
			raise
		try:
			params,_ = curve_fit(sinusoidal_lin,range(len(weeks)),trend,method='trf',p0=(1,np.max(trend)-np.min(trend),1/8,0,(np.max(trend)+np.min(trend))/2,0,0),bounds=sinlbounds,sigma=conf)
			r,m1,f,o,b1,m2,b2 = params[0],params[1],params[2],params[3],params[4],params[5],params[6]
			err1 = np.sqrt(np.sum(np.square(np.array([sinusoidal_lin(tmp,r,m1,f,o,b1,m2,b2) for tmp in range(len(weeks))])-trend)))
			errs.append(err1)
			fits.append(params)
			# print(err1<errf and err1<0.9*err_l,err1<0.9*err_l,errf,err1,0.9*err_l)
			if(err1<errf and err1<explain_factor*err_l):
				fit = "upspike"
				fitf = params
				fittrend_sin = [sinusoidal_lin(tmp,r,m1,f,o,b1,m2,b2) for tmp in range(len(weeks))]
				pred_tmp = [sinusoidal_lin(tmp,r,m1,f,o,b1,m2,b2) for tmp in range(data.shape[1]-predtill,data.shape[1])]
			else:
				fittrend_sin = fittrend_l
		except:
			fittrend_sin = fittrend_l
			pass

		if(i%100==0):
			print("Done",i,"/",data.shape[2])
		pred.append(pred_tmp)
	pred = np.expand_dims(np.array(pred).T,axis=0)
	mae = np.mean(np.abs(pred-true))
	mape = np.mean(np.abs(pred-true)/true)*100
	return mae, mape, pred


def fit_curve_cyc(data,confs,gap=0,predtill=1):
	assert predtill-1<=gap
	true = data[:,-predtill:,:]
	pred = []

	for i in range(data.shape[2]):
		trend = data[0,:-1-gap,i]
		conf = confs[0,:-1-gap,i]

		errs = []
		errf = 5000
		fitf = None
		fittrend = None
		fits = []

		weeks = list(range(trend.shape[0]))
		try:
			params,_ = curve_fit(linear,range(len(weeks)),trend,method='lm',p0=(0,0),sigma=conf)
			# s,i,_,_,_ = linregress(range(len(weeks)),linear,trend,method='lm',p0=(0,0),bounds=([-1/100,0],[1/100,1]),sigma=conf)
			err_l = np.sqrt(np.sum(np.square(np.array([linear(tmp,params[0],params[1]) for tmp in range(len(weeks))])-trend)))
			errs.append(err_l)
			fits.append(params)
			if(err_l<errf):
				fit = "linear"
				errf = err_l
				fitf = params
				fittrend = [linear(tmp,params[0],params[1]) for tmp in range(len(weeks))]
				fittrend_l = [linear(tmp,params[0],params[1]) for tmp in range(len(weeks))]
				pred_tmp = [linear(tmp,params[0],params[1]) for tmp in range(data.shape[1]-predtill,data.shape[1])]
		except:
			raise
		try:
			params,_ = curve_fit(cyclic,range(len(weeks)),trend,method='trf',p0=(np.max(trend)-np.min(trend),2,1/8,0,np.min(trend)),bounds=cycusbounds,sigma=conf)
			m1,k,f,o,b1 = params[0],params[1],params[2],params[3],params[4]
			err1 = np.sqrt(np.sum(np.square(np.array([cyclic(tmp,m1,k,f,o,b1) for tmp in range(len(weeks))])-trend)))
			errs.append(err1)
			fits.append(params)
			# print(err1<errf and err1<0.9*err_l,err1<0.9*err_l,errf,err1,0.9*err_l)
			if(err1<errf and err1<explain_factor*err_l):
				fit = "upspike"
				fitf = params
				errc = err1
				fittrend_c = [cyclic(tmp,m1,k,f,o,b1) for tmp in range(len(weeks))]
				pred_tmp = [cyclic(tmp,m1,k,f,o,b1) for tmp in range(data.shape[1]-predtill,data.shape[1])]
			else:
				fittrend_c = fittrend_l
		except:
			fittrend_c = fittrend_l
			pass
		try:
			params,_ = curve_fit(cyclic,range(len(weeks)),trend,method='trf',p0=(np.max(trend)-np.min(trend),4,1/8,0,np.min(trend)),bounds=cycusbounds,sigma=conf)
			m1,k,f,o,b1 = params[0],params[1],params[2],params[3],params[4]
			err1 = np.sqrt(np.sum(np.square(np.array([cyclic(tmp,m1,k,f,o,b1) for tmp in range(len(weeks))])-trend)))
			errs.append(err1)
			fits.append(params)
			# print(err1<errf and err1<0.9*err_l,err1<0.9*err_l,errf,err1,0.9*err_l)
			if(err1<errf and err1<explain_factor*err_l):
				fit = "upspike"
				fitf = params
				errc = err1
				fittrend_c = [cyclic(tmp,m1,k,f,o,b1) for tmp in range(len(weeks))]
				pred_tmp = [cyclic(tmp,m1,k,f,o,b1) for tmp in range(data.shape[1]-predtill,data.shape[1])]
			else:
				fittrend_c = fittrend_l
		except:
			fittrend_c = fittrend_l
			pass
		try:
			params,_ = curve_fit(cyclic,range(len(weeks)),trend,method='trf',p0=(np.min(trend)-np.max(trend),2,1/8,0,-np.max(trend)),bounds=cycdsbounds,sigma=conf)
			m1,k,f,o,b1 = params[0],params[1],params[2],params[3],params[4]
			err1 = np.sqrt(np.sum(np.square(np.array([cyclic(tmp,m1,k,f,o,b1) for tmp in range(len(weeks))])-trend)))
			errs.append(err1)
			fits.append(params)
			# print(err1<errf and err1<0.9*err_l,err1<0.9*err_l,errf,err1,0.9*err_l)
			if(err1<errf and err1<explain_factor*err_l):
				fit = "upspike"
				fitf = params
				fittrend_c = [cyclic(tmp,m1,k,f,o,b1) for tmp in range(len(weeks))]
				pred_tmp = [cyclic(tmp,m1,k,f,o,b1) for tmp in range(data.shape[1]-predtill,data.shape[1])]
		except:
			fittrend_c = fittrend_l
			pass
		try:
			params,_ = curve_fit(cyclic,range(len(weeks)),trend,method='trf',p0=(np.min(trend)-np.max(trend),4,1/8,0,-np.max(trend)),bounds=cycdsbounds,sigma=conf)
			m1,k,f,o,b1 = params[0],params[1],params[2],params[3],params[4]
			err1 = np.sqrt(np.sum(np.square(np.array([cyclic(tmp,m1,k,f,o,b1) for tmp in range(len(weeks))])-trend)))
			errs.append(err1)
			fits.append(params)
			# print(err1<errf and err1<0.9*err_l,err1<0.9*err_l,errf,err1,0.9*err_l)
			if(err1<errf and err1<explain_factor*err_l):
				fit = "upspike"
				fitf = params
				fittrend_c = [cyclic(tmp,m1,k,f,o,b1) for tmp in range(len(weeks))]
				pred_tmp = [cyclic(tmp,m1,k,f,o,b1) for tmp in range(data.shape[1]-predtill,data.shape[1])]
		except:
			fittrend_c = fittrend_l
			pass

		if(i%100==0):
			print("Done",i,"/",data.shape[2])
		pred.append(pred_tmp)
	pred = np.expand_dims(np.array(pred).T,axis=0)
	mae = np.mean(np.abs(pred-true))
	mape = np.mean(np.abs(pred-true)/true)*100
	return mae, mape, pred



def fit_curve(data,confs,gap=0,predtill=1):
	assert predtill-1<=gap
	true = data[:,-predtill:,:]
	pred = []

	for i in range(data.shape[2]):
		trend = data[0,:-1-gap,i]
		conf = confs[0,:-1-gap,i]

		errs = []
		errf = 5000
		fitf = None
		fittrend = None
		fits = []

		weeks = list(range(trend.shape[0]))
		try:
			params,_ = curve_fit(linear,range(len(weeks)),trend,method='lm',p0=(0,0),sigma=conf)
			# s,i,_,_,_ = linregress(range(len(weeks)),linear,trend,method='lm',p0=(0,0),bounds=([-1/100,0],[1/100,1]),sigma=conf)
			err_l = np.sqrt(np.sum(np.square(np.array([linear(tmp,params[0],params[1]) for tmp in range(len(weeks))])-trend)))
			errs.append(err_l)
			fits.append(params)
			if(err_l<errf):
				fit = "linear"
				errf = err_l
				fitf = params
				fittrend = [linear(tmp,params[0],params[1]) for tmp in range(len(weeks))]
				fittrend_l = [linear(tmp,params[0],params[1]) for tmp in range(len(weeks))]
				pred_tmp = [linear(tmp,params[0],params[1]) for tmp in range(data.shape[1]-predtill,data.shape[1])]
		except:
			raise
		try:
			params,_ = curve_fit(spike,range(len(weeks)),trend,method='trf',p0=(1,np.max(trend)-np.min(trend),2,1/8,0,np.min(trend),0,0),bounds=usbounds,sigma=conf)
			r,m1,w,f,o,b1,m2,b2 = params[0],params[1],params[2],params[3],params[4],params[5],params[6],params[7]
			err1 = np.sqrt(np.sum(np.square(np.array([spike(tmp,r,m1,w,f,o,b1,m2,b2) for tmp in range(len(weeks))])-trend)))
			errs.append(err1)
			fits.append(params)
			# print(err1<errf and err1<0.9*err_l,err1<0.9*err_l,errf,err1,0.9*err_l)
			if(err1<errf and err1<explain_factor*err_l):
				fit = "upspike"
				errf = err1
				fitf = params
				fittrend = [spike(tmp,r,m1,w,f,o,b1,m2,b2) for tmp in range(len(weeks))]
				pred_tmp = [spike(tmp,r,m1,w,f,o,b1,m2,b2) for tmp in range(data.shape[1]-predtill,data.shape[1])]
			# fittrend_s = [spike(tmp,r,m1,w,f,o,b1,m2,b2) for tmp in range(len(weeks))]
		except:
			pass
		try:
			params,_ = curve_fit(spike,range(len(weeks)),trend,method='trf',p0=(1,np.max(trend)-np.min(trend),4,1/8,0,np.min(trend),0,0),bounds=usbounds,sigma=conf)
			r,m1,w,f,o,b1,m2,b2 = params[0],params[1],params[2],params[3],params[4],params[5],params[6],params[7]
			err1 = np.sqrt(np.sum(np.square(np.array([spike(tmp,r,m1,w,f,o,b1,m2,b2) for tmp in range(len(weeks))])-trend)))
			errs.append(err1)
			fits.append(params)
			# print(err1<errf and err1<0.9*err_l,err1<0.9*err_l,errf,err1,0.9*err_l)
			if(err1<errf and err1<explain_factor*err_l):
				fit = "upspike"
				errf = err1
				fitf = params
				fittrend = [spike(tmp,r,m1,w,f,o,b1,m2,b2) for tmp in range(len(weeks))]
				pred_tmp = [spike(tmp,r,m1,w,f,o,b1,m2,b2) for tmp in range(data.shape[1]-predtill,data.shape[1])]
			# fittrend_s = [spike(tmp,r,m1,w,f,o,b1,m2,b2) for tmp in range(len(weeks))]
		except:
			pass
		try:
			params,_ = curve_fit(spike,range(len(weeks)),trend,method='trf',p0=(1,np.min(trend)-np.max(trend),2,1/8,0,-np.max(trend),0,0),bounds=dsbounds,sigma=conf)
			r,m1,w,f,o,b1,m2,b2 = params[0],params[1],params[2],params[3],params[4],params[5],params[6],params[7]
			err1 = np.sqrt(np.sum(np.square(np.array([spike(tmp,r,m1,w,f,o,b1,m2,b2) for tmp in range(len(weeks))])-trend)))
			errs.append(err1)
			fits.append(params)
			if(err1<errf and err1<explain_factor*err_l):
				fit = "downspike"
				errf = err1
				fitf = params
				fittrend = [spike(tmp,r,m1,w,f,o,b1,m2,b2) for tmp in range(len(weeks))]
				pred_tmp = [spike(tmp,r,m1,w,f,o,b1,m2,b2) for tmp in range(data.shape[1]-predtill,data.shape[1])]
		except:
			pass
		try:
			params,_ = curve_fit(spike,range(len(weeks)),trend,method='trf',p0=(1,np.min(trend)-np.max(trend),4,1/8,0,-np.max(trend),0,0),bounds=dsbounds,sigma=conf)
			r,m1,w,f,o,b1,m2,b2 = params[0],params[1],params[2],params[3],params[4],params[5],params[6],params[7]
			err1 = np.sqrt(np.sum(np.square(np.array([spike(tmp,r,m1,w,f,o,b1,m2,b2) for tmp in range(len(weeks))])-trend)))
			errs.append(err1)
			fits.append(params)
			if(err1<errf and err1<explain_factor*err_l):
				fit = "downspike"
				errf = err1
				fitf = params
				fittrend = [spike(tmp,r,m1,w,f,o,b1,m2,b2) for tmp in range(len(weeks))]
				pred_tmp = [spike(tmp,r,m1,w,f,o,b1,m2,b2) for tmp in range(data.shape[1]-predtill,data.shape[1])]
		except:
			pass
		if(i%100==0):
			print("Done",i,"/",data.shape[2])
		pred.append(pred_tmp)
		# plt.figure()
		# plt.plot(range(len(weeks)),fittrend)
		# plt.plot(range(len(weeks)),trend)
		# plt.plot(range(len(weeks)+1-predtill,len(weeks)+1),pred_tmp)
	pred = np.expand_dims(np.array(pred).T,axis=0)
	mae = np.mean(np.abs(pred-true))
	mape = np.mean(np.abs(pred-true)/true)*100
	return mae, mape, pred


################################################################################

import csv

explain_factor = 0.8
trends = np.load('trends.npy')
confs = np.load('confs.npy')

trends = trends[:,:-26]
confs = confs[:,:-26]
gap = 1
predtill = 1
# trends = trends[:,:]
# confs = confs[:,:]
# gap = 26
# predtill = 26

data = np.expand_dims(trends.T,axis=0)
confs = np.expand_dims(confs.T,axis=0)

# data = data[:,:,:10]
# confs = confs[:,:,:10]

mae,mape,pred_mean = mean_pred(data,gap=gap,predtill=predtill)
print("Naive mean:",mae,mape)
mae,mape,pred_last = last_pred(data,gap=gap,predtill=predtill)
print("Naive last:",mae,mape)
mae,mape,pred_exps = exp_smooth_final(data,gap=gap,predtill=predtill)
print("Exponential smoothing",mae,mape)
mae,mape,pred_ar = ar(data,gap=gap,predtill=predtill)
print("Auto Regression",mae,mape)
mae1,mape1,pred_var1 = var(data[:,:,::2],gap=gap,predtill=predtill)
mae2,mape2,pred_var2 = var(data[:,:,1::2],gap=gap,predtill=predtill)
mae = (mae1+mae2)/2
mape = (mape1+mape2)/2
pred_var = np.concatenate([pred_var1,pred_var2],axis=2)

print("Vector Auto Regression",mae,mape)
mae,mape,pred_fcl = fit_curve_lin(data,confs,gap=gap,predtill=predtill)
print("Linear Curve Fit",mae,mape)
mae,mape,pred_fcs = fit_curve_sin(data,confs,gap=gap,predtill=predtill)
print("Sinusoidal Curve Fit",mae,mape)
mae,mape,pred_fcsl = fit_curve_sinl(data,confs,gap=gap,predtill=predtill)
print("Sinusoidal+Lin Curve Fit",mae,mape)
mae,mape,pred_fcc = fit_curve_cyc(data,confs,gap=gap,predtill=predtill)
print("Cyclic Curve Fit",mae,mape)
mae,mape,pred_fc = fit_curve(data,confs,gap=gap,predtill=predtill)
print("Curve Fit",mae,mape)

import matplotlib.pyplot as plt
for i in range(2024):
	plt.figure()
	plt.plot(list(range(trends.shape[1]))[-gap-10:],trends[i,:][-gap-10:],label="True")
	print([trends[i,-2],pred_mean[0,i]])
	plt.plot([trends.shape[1]-2-gap]+list(range(trends.shape[1]-predtill,trends.shape[1])),[trends[i,-2-gap]]+list(pred_mean[0,:,i]),'--',label="mean")
	plt.plot([trends.shape[1]-2-gap]+list(range(trends.shape[1]-predtill,trends.shape[1])),[trends[i,-2-gap]]+list(pred_last[0,:,i]),'--',label="last")
	plt.plot([trends.shape[1]-2-gap]+list(range(trends.shape[1]-predtill,trends.shape[1])),[trends[i,-2-gap]]+list(pred_exps[0,:,i]),'--',label="exps")
	plt.plot([trends.shape[1]-2-gap]+list(range(trends.shape[1]-predtill,trends.shape[1])),[trends[i,-2-gap]]+list(  pred_ar[0,:,i]),'--',label="ar")
	plt.plot([trends.shape[1]-2-gap]+list(range(trends.shape[1]-predtill,trends.shape[1])),[trends[i,-2-gap]]+list( pred_var[0,:,i]),'--',label="var")
	plt.plot([trends.shape[1]-2-gap]+list(range(trends.shape[1]-predtill,trends.shape[1])),[trends[i,-2-gap]]+list(  pred_fcl[0,:,i]),'--',label="linear curve fit")
	plt.plot([trends.shape[1]-2-gap]+list(range(trends.shape[1]-predtill,trends.shape[1])),[trends[i,-2-gap]]+list(  pred_fcs[0,:,i]),'--',label="sinusoidal curve fit")
	plt.plot([trends.shape[1]-2-gap]+list(range(trends.shape[1]-predtill,trends.shape[1])),[trends[i,-2-gap]]+list(  pred_fcsl[0,:,i]),'--',label="sinusoidal+linear curve fit")
	plt.plot([trends.shape[1]-2-gap]+list(range(trends.shape[1]-predtill,trends.shape[1])),[trends[i,-2-gap]]+list(  pred_fcc[0,:,i]),'--',label="cyclic curve fit")
	plt.plot([trends.shape[1]-2-gap]+list(range(trends.shape[1]-predtill,trends.shape[1])),[trends[i,-2-gap]]+list(  pred_fc[0,:,i]),'--',label="curve fit")
	plt.legend()
	plt.show()


