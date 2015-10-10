### This is a machine learning project to differentiate people's voices based on the power embedded in the voice.

from utils import *
from sklearn import cross_validation
from threading import Thread
from sklearn import metrics

TASK_NAME = "speech"

class VoiceClassifier(object):
	def __init__(self,clf=SVC(probability=True),prob=True):
		self.clf = clf
		self.label_to_ind = dict()
		self.prob = prob
	def fit(self,X,y):
		self.clf.fit(X,y)
		self.label_to_ind = dict()
		for i,l in enumerate(self.clf.classes_):
			self.label_to_ind[l]=i
	def predict(self,X):
		pred = self.clf.predict(X)
		all_prob = self.clf.predict_proba(X)
		if self.prob:
			pred_from_prob = np.argmax(all_prob,axis=1)
			pred = map(self.clf.classes_.__getitem__, pred_from_prob)
			prob = np.max(all_prob,axis=1)
		else:
			prob = []
			for i in range(len(pred)):
				prob.append(all_prob[i,self.label_to_ind[pred[i]]])
		return (pred, prob)
	def get_classes(self):
		return self.clf.classes_
	def predict_proba(self,X):
		return self.clf.predict_proba(X)

class FeatureExtractor(object):
	@staticmethod
	def get_features(fs,signal):
		"""
		Return (fund_freq, [features])
		"""
		fund_freq = freq_from_autocorr(signal,fs)
		if fund_freq > 880 or fund_freq < 60:
			return (fund_freq,[])
		f,Pxx_den=scipy.signal.periodogram(signal,fs=fs)
		f_interp = interp1d(f, Pxx_den)
		return (fund_freq,
			   [np.log(1+f_interp(fund_freq*i)) for i in np.arange(1,20,0.5)])


class VoiceManager(object):
	def __init__(self,feature_extractor):
		self.feature_extractor = feature_extractor
		self.load_from_disk()
	def load_from_disk(self):
		self.X = []
		self.y = []
		self.labels = []
		self.freqs = []
		for filename in glob.glob(DATA_DIR+TASK_NAME+"/*.wav"):
			label = get_dataname(filename)
			person,num = label.split("_")
			fs,signal=scipy.io.wavfile.read(DATA_DIR+TASK_NAME+"/%s.wav" % label)
			fund_freq,X = self.feature_extractor.get_features(fs,signal)
			if not X:
				print "Err: %s, %.1f" % (label,fund_freq)
				continue
			else:
				print label,fund_freq
			self.labels.append(label)
			self.freqs.append(fund_freq)
			self.X.append(X)
			self.y.append(person)
	def add(self,fs,signal,label):
		person,num = label.split("_")
		fund_freq,features = self.feature_extractor.get_features(fs,signal)
		self.X.append(features)
		self.y.append(person)
		self.labels.append(label)
		self.freqs.append(fund_freq)
	def get_snapshot(self,full=False):
		if full:
			return self.X, self.y, self.labels, self.freqs
		else:
			return self.X, self.y


class Recorder(object):
	def __init__(self,try_once=False):
		self.try_once = try_once
		self.p = pyaudio.PyAudio()
		self.threshold = 500
		self.chunk_size = 1024
		self.format = pyaudio.paInt16
		self.sample_width = self.p.get_sample_size(self.format)
		self.fs = 44100
	def is_silent(self, snd_data):
		"Returns 'True' if below the 'silent' threshold"
		return np.mean(map(abs,snd_data)) < self.threshold
	def normalize(self,snd_data):
		"Average the volume out"
		MAXIMUM = 16384
		times = float(MAXIMUM)/max(abs(i) for i in snd_data)
		r = array('h')
		for i in snd_data:
			r.append(int(i*times))
		return r
	def record(self,min_sec=0.05):
		"""
		Record a word or words from the microphone and 
		return the data as an array of signed shorts.
		"""
		stream = self.p.open(format=FORMAT, channels=1, rate=RATE,
			input=True, output=True,
			frames_per_buffer=CHUNK_SIZE)
		num_silent = 0
		snd_started = False
		r = array('h')
		print "Go!"
		num_periods = 0
		while 1:
			# little endian, signed short
			snd_data = array('h', stream.read(CHUNK_SIZE))
			if byteorder == 'big':
				snd_data.byteswap()
			silent = self.is_silent(snd_data)
			if silent and snd_started:
				if num_periods <= RATE / CHUNK_SIZE / 2 * min_sec :
					print "Too short, resampling"
					if self.try_once: return
					snd_started = False
					r = array('h')
					num_periods = 0
					continue
				else:
					break
			elif silent and not snd_started: # hasn't started yet
				continue
			elif not silent and snd_started: # okay
				r.extend(self.normalize(snd_data))
				num_periods += 1
			else: # sound just started
				print "Start recording"
				snd_started = True
		print "Finish"
		r = r[:-CHUNK_SIZE]
		stream.stop_stream()
		stream.close()
		return r
	def __del__(self):
		self.p.terminate()
	def findmax(self,label):
		largest = -1
		for filename in glob.glob(DATA_DIR+TASK_NAME+"/%s_*.wav" % label):
			largest = max(largest,int(re.findall(DATA_DIR+TASK_NAME+"/%s_(\d+).wav" % label,filename)[0]))
		return largest
	def save(self,signal,label):
		"Records from the microphone and outputs the resulting data to 'path'"
		signal = pack('<' + ('h'*len(signal)), *signal)
		next_id = self.findmax(label) + 1
		recording_name = "%s_%d" % (label,next_id)
		wf = wave.open(DATA_DIR+TASK_NAME+"/%s.wav" % recording_name, 'wb')
		wf.setnchannels(1)
		wf.setsampwidth(self.sample_width)
		wf.setframerate(self.fs)
		wf.writeframes(signal)
		wf.close()
		return recording_name

def cross_validate(data_manager,voice_clf,shuffle=False,n_folds=10,n_trials=1,verbose=0):
	if n_trials > 1:
		all_scores = []
		for i in xrange(n_trials):
			scores = cross_validate(data_manager,voice_clf,shuffle=shuffle,n_folds=n_folds)
			if verbose >= 1: print "CV Trial %d: Mean Accuracy: %.3f" % (i,np.mean(scores))
			all_scores += scores
		if verbose >= 1: print "Mean Accuracy: %.3f" % np.mean(all_scores)
		return all_scores
	n = len(data_manager.y)
	kf = cross_validation.KFold(n, shuffle=shuffle, n_folds=n_folds)
	count = 1
	accuracy_scores = []
	accuracy_scores_2 = []
	X,y = np.array(data_manager.X), np.array(data_manager.y)
	for train_index, test_index in kf:
		if verbose >= 2:
			print "Cross Validating: %d/%d" % (count, n_folds),
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = y[train_index], y[test_index]
		voice_clf.fit(X_train,y_train)
		y_pred,y_score = voice_clf.predict(X_test)
		accuracy_score = metrics.accuracy_score(y_test,y_pred)
		accuracy_scores.append(accuracy_score)
		count += 1
		if verbose >= 2:
			print "Accuracy = %.3f" % accuracy_score
	if verbose >= 1:
		print "Mean Accuracy: %.3f" % np.mean(accuracy_scores)
	return accuracy_scores



