from __future__ import division
import numpy as np

# Hidden Markov Model
# @see http://en.wikipedia.org/wiki/Hidden_Markov_model
class HMM(object) :
	def __init__(self) :
		self.initial = {}
		self.transitions = {}
		self.emissions = {}

	# Expectation Maximization using the Forwards-Backwards algorithm
	# used to predict the next state
	# @see http://en.wikipedia.org/wiki/Forward%E2%80%93backward_algorithm
	def fwd_bkwd(self,states,transitions,emissions,observations) :
		fwd,bkwd,gamma = [],[],[]

		# start and end with a uniform probability of all events, 
		# meaning that we don't know what state the system begins in
		# and that the system ends in a particular state
		fwd.append( np.array( [ 0.5 ] * len(states) ) )
		bkwd.append( np.array( [ 1 ] * len(states) ) )

		for observation in observations :
			
			# forward probabilities
			observation = np.diagflat(observation)
			m = np.dot(np.dot(observation,np.transpose(transitions)),fwd[-1])
			m /= sum(m)
			fwd.append(m)

			# backward probabilities
			m2 = np.dot(np.dot(np.transpose(transitions),observation),bkwd[-1])
			m2 /= sum(m2)
			bkwd.append(m2)

		for i in range(0,len(fwd)) :
			g = []
			for j in range(0,len(fwd[i])) :
				g.append(fwd[i][j] * bkwd[ - ( i - ( len(fwd) - 1 ) ) ][j])
			g /= sum(g)
			gamma.append(g)

		return fwd, bkwd, gamma

	# used in determining the most likely sequence of states
	# given a sequence of observations
	def viterbi(self) :
		print "TODO"
