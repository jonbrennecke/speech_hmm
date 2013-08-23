# Signal processing library

from __future__ import division
import numpy as np

# Mel-Frequency Cepstrum Coefficients (MFCC) algorithm 
# similar to the algorithm described by the European Telecommunications Standards Institute
# @see - http://www.etsi.org/deliver/etsi_es/201100_201199/201108/01.01.03_60/es_201108v010103p.pdf
# @see - http://en.wikipedia.org/wiki/Mel-frequency_cepstrum
# @see - http://www.practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/
def mfcc(signal) :

	# some signal filtering
	offset_free_signal = notch_filter(signal)
	pre_signal = preemphasis_filter(offset_free_signal)
	windowed_signal = hamming_window(pre_signal)

	# convert signal (time-domain) to frequency domain
	transformed_signal = fft(windowed_signal)

	# the human ear has high-frequency resolution at low-frequency parts of the spetrum
	# and low frequency resolution at high parts of the spectrum
	# thus to more accurately mimick the frequency  resolution of the human ear
	# we need to convert a frequency into the mel-scale
	fbank = mel_filter(transformed_signal)
	c = cepstral_coeffs(fbank)

	# energy measure
	squares = [ n*n for n in signal ]
	sigma = np.log(sum(squares)) 
	logE = sigma if sigma > -50 else -50

	# the final feature vector consists of 14 coefficients
	# the log-energy coefficient and the 13 cepstral coefficients
	c.append(logE)
	return c

# Transforming the filter bank with natural 
# logarithm and DCT yields the cepstral coefficients
def cepstral_coeffs(fbank) :
	log_fbank = [ np.log(n) if np.log(n) > -50 else -50 for n in fbank ]
	
	# Discrete Cosine Transform
	c = []
	for i in range(0,13) :
		p = [ log_fbank[j-1] * np.cos(((np.pi*i)/23)*(j-0.5)) for j in range(1,24) ]
		c.append(sum(p))
	return c

# low frequency elements are ignored.
# the most useful frequency band lies between 64Hz and half of the sampling frequency
# the output of the mel filter is the weighted sum of the FFT spectrum values (bin[i])
# in each band
def mel_filter(spectrum,n=24) :

	# mel scale function
	def mel(x) :
		return 2595 * np.log10(1 + (x/700))

	# inverse mel function
	def inv_mel(x) :
		return 700 * ( np.exp(x/1127) - 1 )

	fstart = 64 # 64 Hz  
	fend = 4000 # 4 kHz (half of sampling frequency)
	mel_points = []
	mel_points.append(mel(fstart)) # upper bound of frequency (limited to half of the sampling frequency)
	mel_points.append(mel(fend)) # lower bound of frequency
	d = (mel_points[1]-mel_points[0])/(n+1) # unit distance
	points = [ i*d+mel(fstart) for i in range(1,n+1) ] 

	# now compute n points linearly between the upper and lower bound
	# and return them to the frequency domain
	mel_points.extend(points)
	mel_points = sorted(mel_points)
	c = [ int((inv_mel(n)/fend)*len(spectrum)) for n in mel_points ]

	fbank = []
	for k in range(0,23) :
		part1, part2 = 0,0
		for i in range(int(c[k]),int(c[k+1])) :
			part1 += ((i - c[k] + 1)/(c[k+1]-c[k]+1))*spectrum[i]
		for i in range(c[k+1]+1,c[k+2]) :
			part2 += (1 - (i-c[k+1])/(c[k+2]-c[k+1]+1))*spectrum[i]
		fbank.append(part1+part2)

	return fbank

# a hamming window of length len(signal) is applied to the signal
def hamming_window(signal) :
	windowed_signal = []
	for i in range(0,len(signal)) :
		windowed_signal.append(( 0.54 - 0.46*np.cos((2*np.pi*(i  - 1 ))/(len(signal)-1)))*signal[i])
	return windowed_signal

# pre-emphasis filter applied to the framed offset-corrected input signal
def preemphasis_filter(signal) :
	pre_signal = []
	for i in range(0,len(signal)) :
		pre_signal.append( signal[i] - ( 0.97 * signal[i-1] ) )
	return pre_signal

# notch filter to remove the DC component of the signal waveform
def notch_filter(signal) :
	offset_free_signal = [ signal[0] ] * len(signal)
	for i in range(1,len(signal)) :
		offset_free_signal[i] = signal[i] - signal[i-1] + ( 0.999 * offset_free_signal[i-1] )
	return offset_free_signal

# Discrete (Fast) Fourier Transform
def fft(signal) :
	bins = []
	for k in range(0,len(signal)) :
		fft_val = [ signal[n] * np.exp((-2j*n*k*np.pi)/len(signal)) for n in range(0,len(signal)) ]
		bins.append( abs( sum( fft_val ) ) )
	return bins
