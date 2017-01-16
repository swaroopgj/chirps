'''
Created on 2 Nov 2016

@author: jl10015
'''

import os
import librosa
import numpy as np


def framing(filename):


    sound_clip,s = librosa.load(filename,sr=11025*2)
    
    
    
    
    mel =librosa.feature.mfcc(sound_clip,n_mfcc=28)
    mfcc_delta = librosa.feature.delta(mel)
    mfcc_delta2 = librosa.feature.delta(mel, order=2)
    
    
   
    mel=np.resize(mel, (28,420))
    mfcc_delta=np.resize(mfcc_delta, (28,420))
    mfcc_delta2=np.resize(mfcc_delta2, (28,420))
    #print mel[:,1:10]
    
    
    '''print mel.shape
    print mfcc_delta.shape
    print mfcc_delta2.shape'''
    
    mel2=np.zeros((28,28))
    mfcc_delta2=np.zeros((28,28))
    mfcc_delta22=np.zeros((28,28))
    #print mel2[:,0].shape
    
    #mel2[:,0]=np.mean(mel[:,15*0:15*(0+1)], axis=1)
    
    
    for i in xrange(28):
        mel2[:,i]= np.mean(mel[:,15*i:15*(i+1)], axis=1)
        mfcc_delta2[:,i]= np.mean(mfcc_delta[:,15*i:15*(i+1)], axis=1)
        mfcc_delta22[:,i]= np.mean(mfcc_delta2[:,15*i:15*(i+1)], axis=1)
    #print mel2
    mel2=mel2.reshape((1,28*28))
    mfcc_delta2=mfcc_delta2.reshape((1,28*28))
    mfcc_delta22=mfcc_delta22.reshape((1,28*28))
    
    #exit()
    #plt.imshow(mel,interpolation=None,aspect='auto')
    #plt.show()
    '''
    window_size= sound_clip.size / (2*nbrFrames)
    
    
    prefeat=np.zeros((2*nbrFrames,2*window_size ))
    
    for i in xrange(nbrFrames):
            prefeat[2*i]=sound_clip[window_size*i:window_size*(i+2)]
            prefeat[2*i+1]=sound_clip[window_size*(i+1):window_size*(i+3)]
            
    
    features=np.zeros((2*nbrFrames,120))        
    for i in xrange(2*nbrFrames):                
        mfccs = np.mean(librosa.feature.mfcc(y=prefeat[i], sr=s, n_mfcc=40).T,axis=0)
        mfcc_delta = librosa.feature.delta(mfccs)
        mfcc_delta2 = librosa.feature.delta(mfccs,order=2)
        features[i]=np.concatenate((mfccs,mfcc_delta,mfcc_delta2),axis=0)
    '''
    
    #print mel2.shape
    f = open("fmel2828.txt", "a")
    np.savetxt(f, mel2)
    f.close()
    f1 = open("fmfccDelta2828.txt", "a")
    np.savetxt(f1, mfcc_delta2)
    f1.close()
    f2 = open("fmfccDelta22828.txt", "a")
    np.savetxt(f2, mfcc_delta22)
    f2.close()

    

       
    
    
if __name__=='__main__':
    
    #framing('/home/jl10015/workspace/Birdy/wav/0a42af88-f61a-4504-9ba2.wav')
    
    k=0 
    for fn in os.listdir('/home/jl10015/workspace/Birdy/wav0'):
        
        if k <6700:
            framing('/home/jl10015/workspace/Birdy/wav0/'+fn)
            
           
            k=k+1
            print k



    
    
    
    
    
    
