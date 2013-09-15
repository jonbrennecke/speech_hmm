from __future__ import division
import numpy as np
import re

# a SOM is the training function we'll use to initialize 
# our neural network
class SelfOrganizingMap() :
	def __init__(self,network) :
		self.network = network

	# update step of the training algorithm
	def update(self,inputs,index,period) :

		# distance loop
		allweights = self.network.getWeights()
		dist, modinputs = [], []
		for i in range(0,len(allweights)) :
			modinputs.append(inputs)
			wsum = self.layerSum(inputs,allweights[i])
			for j in range(0,len(allweights[i])) :

				# the n-dimensional Euclidean distance is a measure of
				# the similarity between the input vector and the neuron's weights
				dist.append( [ self.distance( allweights[i][j], inputs ), self.network.getNeuron(i,j) ] )

			inputs = [ self.network.sigmoid(s) for s in wsum ]

		# sort
		dist = np.array(dist)
		sort = dist[:,0].argsort()   
		neurons = dist[:,1][sort]
		sortedDist = dist[:,0][sort]

		# the best matching unit (BMU) has the smallest distance
		bmu = neurons[0]

		# Since the distance between the BMU and the input vector is
		# assumed to be relatively small, it's simpler to reuse
		# the sorted list of neurons to find the nearest neighbors of the BMU.
		radius = self.neighborhood(index,period,sortedDist[-1])
		for i in range(0,len(sortedDist)) :
			if radius > sortedDist[i] :
				self.adjustWeights(
					neurons[i],
					index,
					period,
					sortedDist[-1],
					sortedDist[i],
					modinputs )

	# returns a gaussian distribution
	def gaussian(self,dist,sigma) :
		return np.exp((- dist * dist ) / ( 2 * sigma * sigma) )

	# exponential decay function to calculate the radius
	# of a neighborhood around the input vector
	def neighborhood(self,index,period,factor) :
		return factor*np.exp(-index/period)

	# adjust the weight of a neuron
	def adjustWeights(self,neuron,index,period,factor,dist,inputs) :
		l = self.neighborhood(index,period,0.1)
		sigma = self.neighborhood(index,period,factor)
		gauss = self.gaussian(dist,sigma)
		v = self.searchForSize(inputs,len(neuron.weights))
		neuron.weights = neuron.weights + ( gauss * l * ( v - neuron.weights ) )

	# search a 2D list for a list of a particular length, return the first one found
	def searchForSize(self,vector,size) :
		for v in vector :
			if len(v) == size :
				return v
		raise Exception('sub-array of size %s could not be found' % size) 

	def layerSum(self,inputs,layer) :
		wsum = []
		for neuron in layer :
			wsum.append(sum(neuron * inputs))
		return wsum

	def distance(self,v1,v2) :
		return np.sqrt(sum(np.square(v1 - v2)))

class Network() :
	def __init__(self,ninputs,noutputs) :

		# __init__ from file
		# always try opening the ann.log file first
		try :
			with open('ann.log','r') as f :
				doc = f.read()
				layers = self.parseFile(doc)
				self.layers = []
				for layer in layers :
					weights = [ Neuron(w) for w in layer ]
					self.layers.append( NeuronLayer( **{'neurons' : weights } ) )

				self.ninputs = self.layers[0].ninputs
				self.noutputs = self.layers[-1].ninputs + 2
				self.numlayers = abs( self.ninputs - self.noutputs )

		except IOError :

			# __init__ from input
			self.ninputs = ninputs
			self.noutputs = noutputs
			self.numlayers = abs( noutputs - ninputs )	
			self.layers = [ NeuronLayer( **{ 'nneurons' : ninputs+i+1, 'ninputs' : ninputs+i }) for i in range(0,self.numlayers) ]

	# load the neural network from a file
	def parseFile(self,doc) :
		pattern = re.compile('([]]|[[])')
		result = re.split(pattern,doc)
		result = filter(None,result)
		layers = [ list() ]
		for i in range(0,len(result)) :

			if result[i] == '[' and result[i+1] != '[' and len(result[i+1]) > 2 :
				l = re.split(re.compile('\s+'),result[i+1])
				l = [ s.strip(',') for s in l ]
				l = filter(None,l)
				l = map(float,l)
				layers[-1].append(l)

			try :
				if result[i+1] == '[' and result[i] == '[' :
					layers.append(list())
			except IndexError :
				pass

		return layers[1::]

	# create an 'ann.log' file that serves as a dump of the neural network's current state
	def createFile(self) :
		f = open('ann.log', 'w')
		for layer in self.layers :
			weights = layer.getWeights()
			f.write( str(weights) )
		f.close()

	def update(self,inputs) :
		if len(inputs) != self.ninputs :
			return 

		for layer in self.layers :
			wsum = layer.sum(inputs)
			inputs = [ self.sigmoid(s) for s in wsum ]

		return inputs

	def getWeights(self) :
		return np.array([ layer.getWeights() for layer in self.layers ])

	def sigmoid(self,x) :
		return 1/(1+np.exp(-x))

	def getNeuron(self,i,j) :
		return self.layers[i].neurons[j]

class NeuronLayer() :
	def __init__(self,*args,**kwargs) :
		if kwargs.get('neurons') :
			neurons = kwargs.get('neurons')
			self.ninputs = len(neurons)
			self.neurons = neurons

		else :
			self.ninputs = kwargs.get('ninputs')
			self.neurons = [ Neuron(self.ninputs) for i in range(0,kwargs.get('nneurons')) ]

	def __len__(self) :
		return self.ninputs

	def sum(self,inputs) :
		wsum = []
		for neuron in self.neurons :
			wsum.append(sum(neuron.weights * inputs))
		return wsum

	def getWeights(self) :
		return np.array([ neuron.weights for neuron in self.neurons ])

class Neuron() :
	def __init__(self,inputs) :

		if type(inputs) == int :
			self.weights = np.random.random(inputs)
		
		if type(inputs) == list :
			self.weights = np.array(inputs)

	def __len__(self) :
		return len(self.weights)
