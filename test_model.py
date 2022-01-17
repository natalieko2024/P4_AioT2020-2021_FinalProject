def getResult(audioDirectory, filename):
	import tensorflow as tf
	import cv2
	import os
	import numpy as np
	import pandas as pd
	import librosa
	from librosa import display
	from matplotlib import pyplot as plt
	from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

	# Load a model
	model.load('/Users/natalieko/Desktop/0.3514')

	os.chdir(audioDirectory)
	y, sr = librosa.load(filename, sr=32000, mono=True, duration=20)
	melspec = librosa.feature.melspectrogram(y, sr=sr, n_mels = 128)
	melspec = librosa.power_to_db(melspec).astype(np.float32)

	window_size = 1024
	window = np.hanning(window_size)
	stft  = librosa.core.spectrum.stft(y, n_fft=window_size, hop_length=512, window=window)
	out = 2 * np.abs(stft) / np.sum(window)

	fig = plt.Figure()
	canvas = FigureCanvas(fig)
	ax = fig.add_subplot(111)
	p = librosa.display.specshow(librosa.amplitude_to_db(out, ref=np.max), ax=ax, y_axis='log', x_axis='time')
	fig.savefig(filename + '.png')

	test_image = np.asarray(Image.open(audioDirectory).convert("RGB").resize((401, 262)))
	test_image = np.array(test_image).reshape(401, 262, 3)
	prediction = model.predict({test_image})

	return prediction