from __future__ import division
import os, wave, struct
import signallib
import numpy as np
import ann


if __name__ == '__main__':

	index = 0
	chunk = 256
	n = ann.Network(14,30)
	mp = ann.SelfOrganizingMap(n) 

	try :
		files = os.listdir("../wav")
		for filename in files :
			try :
				wav = wave.open("../wav/" + filename)
				bin = wav.readframes(wav.getnframes())
				data = []
				for i in range(0,len(bin)) :
					if i%2 != 0 :
						continue

					# convert the .wav to decimal
					data.append(struct.unpack("<h", bin[i:i+2])[0])

				rate,chunk = (wav.getframerate(),256)
				duration = (len(data)/rate)*1000 # duration of stream in ms

				# feature vectors are computed every 10ms
				# in an overlapping analysis window of 30ms
				featureVector = []
				for i in range(0,int(duration/10)) :
					frame = data[i*10:(i*10)+30]
					mfcc = signallib.mfcc(frame)
					output = n.update(mfcc)
					mp.update( mfcc,index,len(files) * 3 )
					index += 1

				name = filename.split('.')[0]
				print name
				# f = open('../lib/' + name + '.log', 'w')
				# f.write(str(mfcc))
				# f.close()

			except IOError:
				pass

	except OSError :
		pass

	# once the Self Organizing Map (SOM) has finished the training loop on every file in the .wav directory
	# save a dump of the neural network's current state
	n.createFile()
