import os, librosa
import numpy as np
import cPickle as pickle

def features(filename, mels=False, logamp=False, n_mfcc=128):
    y, sr = librosa.load(filename, sr=22050)
    #librosa.feature.chroma_cqt(y, sr, fmin=20, n_chroma=12, n_octaves=9)
    # 128x217
    if mels:
        S = librosa.feature.mfcc(y, n_mels=128, sr=sr, n_mfcc=n_mfcc, hop_length=1024, n_fft=1024*4,
                                           fmin=20, fmax=10000)
    else:
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, hop_length=1024, n_fft=1024*4,
                                           fmin=20, fmax=10000)
        if logamp:
            S = librosa.logamplitude(S, ref_power=np.median)
    return S.reshape(1, -1)


def test_data(path='test_data', mels=True):
    song_specs = []
    song_names = []
    count = 0
    for fn in sorted(os.listdir(path)):
        song_specs.append(features(path+'/'+fn, mels=mels, n_mfcc=20))
        song_names.append(fn)
        count += 1
        if count % 100 == 0:
            print "Processed %d" % count
    song_specs = [s.astype('float16') for s in song_specs]
    data = {'data': song_specs, 'names': song_names}
    out_fname = 'test_data_mfcc.pkl' if mels else 'test_data_melspec.pkl'
    with open(out_fname, 'wb') as outfile:
        pickle.dump(data, outfile, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    mels = True
    song_specs = []
    song_names = []
    count = 0
    for fn in sorted(os.listdir('wav')):
        song_specs.append(features('wav/'+fn, mels=mels))
        song_names.append(fn.split('.wav')[0])
        count += 1
        if count % 100 == 0:
            print "Processed %d" % count
    song_specs = [s.astype('float16') for s in song_specs]
    labels1 = np.genfromtxt('ff1010bird_metadata.csv',
                            dtype=[('filename', 'S32'), ('hasbird', 'i1')], delimiter=',',
                            names=True)
    labels2 = np.genfromtxt('warblrb10k_public_metadata.csv',
                            dtype=[('filename', 'S32'), ('hasbird', 'i1')], delimiter=',',
                            names=True)
    labels = np.hstack([labels1, labels2])
    labels = {k: v for k, v in labels}
    data = {'data': song_specs, 'names': song_names, 'labels': labels,
            'format': '128 by X points approximately 5hz'}
    out_fname = 'song_data_mfcc.pkl' if mels else 'song_data_melspec.pkl'
    with open(out_fname, 'wb') as outfile:
        pickle.dump(data, outfile, pickle.HIGHEST_PROTOCOL)






