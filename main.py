#!/usr/bin/env python
from __future__ import division
from ctypes import *
from datetime import datetime
import pyaudio, sys, threading, struct
from PyQt4 import QtGui, QtCore
import numpy as np

from canvas import Canvas, Mouse
import signalPy
from hmm import HMM
import audio

class Speech_Recognition(HMM) :
	def __init__(self,path) :
		super(Speech_Recognition,self).__init__()

# The back end analytics object
class Analytics() :
	def __init__(self,parent) :
		self.parent = parent

	def handle_mic_event(self,event) :
		stream,rate,chunk = event.unpack()
		# rate * chunk

		duration = (len(stream)/rate)*1000 # duration of stream in ms
		
		# feature vectors are computed every 10ms
		# in an overlapping analysis window of 30ms
		featureVector = []
		for i in range(0,int(duration/10)) :
			frame = stream[i*10:(i*10)+30]
			featureVector.append(signalPy.mfcc(frame))

		print featureVector
		# print 'here'

		# print featureVector
	

# main application window
class Application(QtGui.QWidget) :
	def __init__(self) :
		super(Application, self).__init__()
		self.analytics = Analytics(self)
		self.mic = audio.Microphone(self,rate=8000,chunk=256,handler=self.analytics)
		self.size = (50,100)
		self.installEventFilter(self)
		self.initUI()

	def initUI(self):
		self.setGeometry(300,300,self.size[0],self.size[1])
		self.setWindowTitle('Test')
		self.recordButton = QtGui.QPushButton('Record', self)
		self.recordButton.clicked.connect(self.mic.startStream)
		self.pauseButton = QtGui.QPushButton('Pause', self)
		self.pauseButton.clicked.connect(self.mic.pause)

		# horizontal box
		hbox = QtGui.QHBoxLayout()
		# hbox.addStretch(1)
		hbox.addWidget(self.recordButton)
		hbox.addWidget(self.pauseButton)
		self.setLayout(hbox)

		self.show()

	def eventFilter(self,object,event) :
		return False

if __name__ == '__main__': 

	app = QtGui.QApplication(sys.argv)
	ex = Application()
	sys.exit(app.exec_())
