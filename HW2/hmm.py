import numpy as np

class HMM(object):
	"""This class defines an Hidden Markov Model.
	 An HMM is defined as a tuple (X, Z, P, O) where:
	 X - is the set of possible states.
	 Z - the set of possible observation.
	 P - is the transition probability matrix.
	 O - is the obsertation/emission probability matrix. 
	"""
	def __init__(self, X, Z, P, O, inicial_dist=None):
		""" Create a HMM. If no initial distribution is given it will assume that it is equal for each state."""
		self.X = np.array(X)
		self.Z = np.array(Z)
		self.P = np.array(P)
		self.O = np.array(O)

		if inicial_dist:
			self.inicial_dist = np.array(inicial_dist)
		else:
			self.inicial_dist = np.array([1/len(X)]*len(X))

	def state2ix(self, state_name):
		for i in range(self.X.shape[0]):
			if self.X[i] == state_name:
				return i

	def emission_distribution(self, e):
		""" Maps an observation/emission string into the correct distribution for that emission. """
		for i in range(0, len(self.Z)):
			if e == self.Z[i]:
				return self.O[:,i]
		return None

	def viterbi(self, seq):
		""" Viterbi Algorithm. Returns a list of states index."""
		# initialization
		m = np.diag(self.emission_distribution(seq[0])).dot(self.inicial_dist)
		I = [[]]*(len(seq)-1)
		# cycle over the sequence
		for i in range(1, len(seq)):
			I[i-1] = np.argmax(self.P.T.dot(np.diag(m)), axis=1)
			m = np.diag(self.emission_distribution(seq[i])).dot(np.amax(self.P.T.dot(np.diag(m)), axis=1))
		# backtrack
		# current variable will be initialized as the most probable final state and it will backtrack until the first state.
		current = np.argmax(m) 
		states = [current,]
		for i in range(len(I)-1, -1, -1):
			current = I[i][current]
			states.append(current)
		states.reverse()
		return states

	def forward(self, seq):
		"""" Forward algorithm. Returns a vector with the probability of each state given an observed sequence. """
		alfas = [self.inicial_dist]
		for i in range(0, len(seq)):
			alfas.append((np.diag(self.emission_distribution(seq[i])).dot(self.P.T)).dot(alfas[-1]))
		return alfas

	def forward_backward(self, seq, final_state=None):
		# forward
		alfas = self.forward(seq)

		# beta initialization
		initial_beta = self.P[:, self.state2ix(final_state)] * self.emission_distribution(seq[-1]) \
		if final_state else np.ones(self.X.shape[0])
		# backward
		betas = [initial_beta]
		for i in range(len(seq)-1, -1, -1):
			betas.append(self.P.dot(np.diag(self.emission_distribution(seq[i]))).dot(betas[-1]))
		betas.reverse()

		# combine forward and backward results.
		results = []
		for i in range(len(alfas)):
			results.append(self.norm(alfas[i] * betas[i]))
		return results


	def norm(self, vec):
		""" Auxiliar function to normalize a vector. Usefull to normalize the forward result."""
		return vec/np.sum(vec)

def main():

	P = [[0.7,  0.2,  0.1],
		 [0.3, 0.5,  0.2],
		 [0.2, 0.3, 0.5]]

	O = [[0.4, 0.4, 0.1, 0.1], 
		 [0.5, 0.1, 0.2, 0.2], 
		 [0.1, 0.1, 0.3, 0.5]]

	X = ["Sunny", "Windy", "Rainy"]
	Z = ["Surf", "Beach", "Video-Game" ,"Study"]

	# Since we know that the weather in October 7 was Rainy, the initial distribution is:
	initial_dist = [0, 0, 1]
	last_week_obs = ["Video-Game", "Study", "Study", "Surf", "Beach", "Video-Game", "Beach"]
	# Initialize Model
	hmm = HMM(X, Z, P, O, initial_dist)
	states = hmm.viterbi(last_week_obs)
	path = "Weather for the past week:\n"
	for i in range(len(states)-1):
		path += X[states[i]] + " - "
	path += X[states[i+1]]
	print (path)

	decoded_states = ""
	min_risk_decoding = hmm.forward_backward(last_week_obs, "Sunny")
	for i in range(1, len(min_risk_decoding)-1):
		decoded_states += X[np.argmax(min_risk_decoding[i])] + " - "
	decoded_states += X[np.argmax(min_risk_decoding[i])]
	print (decoded_states)

if __name__ == "__main__":
	main()