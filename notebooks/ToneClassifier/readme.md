This folder contains all the code Huw Day wrote for the Seedcorn project (April 2024 - June 2024).
Three main sections of code, order of code is intentional (pre-processing etc.)

# 1) Standalone Colour Map Classifier (proof of concept)

## colourmapclassifier.ipynb
- This code was early exploration of the ISIC data, as well as initial attempt of a colour map classifier.
- The early approach for classifying skintone didn't use neural networks.
- Instead, the approach was to convert pixels from RGB values into ITA values.
- From there, a histogram type approach sorts the ITA values into bands and then the modal value is the band.
- At the end, you can use a variety of classifiers to use modal value (a single value) to try and predict skin tone.
- Instead of predicting Fitzpatrick Skintone from 1-6, we specified 1-2 as light skin tone and 3-6 as dark skin tone. This variable is called "ToneBinary".
- Some code at the end evaluates how good the classifiers and in particular notes that the optimal performance is about in line with the class imbalance of the dataset.
- This will be a common theme with other skin tone classifiers.

# 2) Neural Network taking original ISIC images and classifying light or dark skin tone

## TestTrainSplit.py
- Very simple bit of code that does the Test Train Split of the ISIC images and saves them as separate csv files
- Can be altered to fix a random state or vary ratio of test to train.
- Currently doesn't allow for validation outside of the neural network.

## ClassWeight.py
- Works out the class imbalance in case you want to use this in your code.

## CNNTrialDataset.py 
- Makes the classes which are needed for training the neural net in CNNTrial.py
- Data preparation with various potential augmentations on the images for robustness
  
## CNNTrial.py
- Fairly out the box resent 50 neural net
- Some attempts to deal with the massive class imbalance
- Set up to track model with weights and biases

## [Weights and Biases Project page](https://wandb.ai/disco_huw/SkinToneSeedcorn?nw=nwuserhuwday)

# 3) Neural Network taking ITA arrays of ISIC images and classifying light or dark skin tone

## ImagesToITAMatrices.py
- Pre-processing for a secondary approach, where rather than feeding a neural network raw images, we convert images into arrays of ITA values.
- Input ISIC images (which have associated Meta data)
- This code converts those ISIC images into matrices where each pixel is the ITA angle
- Idea is to feed the ITA matrices into a NN to classify skin tone

## To be continued
- Next steps would be to write/use existing neural network code to load in these .npy arrays
- From there, train a skin tone classifier but using ITA matrices as an input
- Make sure to change your data loading to account for the reduced dimension (image rgb arrays are 3x as big as the ITA arrays)
- Perform transformations to make sure all arrays are the same size, not sure the best way to do this. Could even consider doing this before doing the ITA transformations.
- Also might want to add random rotations or flips for robustness. 
  
