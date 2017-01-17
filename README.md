# chirps - Bird Audio Detection Challenge
<http://machine-listening.eecs.qmul.ac.uk/bird-audio-detection-challenge/>

## Features 
We used 2 sets of features: 1) MFCCs and 2) Spectrum in melscale (melspectrum),
both computed using the package librosa <http://librosa.github.io/>.

For MFCCs, we used 20 coefficients sampled at ~20Hz. This gave us 20x200 features for 
a 10s clip after clipping the first 5 samples and keeping 200 for 10s clip.

For spectrum, we use power in high frequency range from approximately 1500-10000Hz at
~20Hz sampling rate. T

## Learning algorithm
We used a convolution neural network (CNN) coded in tensorflow with 3 convolution layers and 2
fully-connected layers for bothe feature sets separately.
Our intuition is that we (humans) could look at the spectrogram and could identify signatures
of bird songs (even without listening to them), and that means we can leverage a visual model
to detect bird songs in the spectrum.

We trained separate CNNs on mfcc and melspectrum, and combined the probabilities by picking
the maximum probability for detecting a bird song across these networks.
Even with a single CNN based on 20 MFCCs sampled every 5 ms, our trained network got ~83.3%
accuracy. By combining 2 trained CNNs per feature set (a total of 4), we got upto ~84.5%.
