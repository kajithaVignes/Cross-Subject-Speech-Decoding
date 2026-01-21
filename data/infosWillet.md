# Willett dataset

one line of the dataset = one trial = neural activity for one sentence, recorded every 20ms (can be of different lenghts!!)

## What is a trial?
* Each trial contains data from the "go" period of a sentence production instructed delay experiment. 
* On each trial, the participant first saw a red square with a sentence above it. 
* Then, when the square turned green, she either attempted to speak that sentence normally, or attempted to speak the sentence without vocalizing ("mouthing"), depending on the day. 
* When she finished, she pressed a button on her lap that triggered the beginning of the next trial. 

## Feature extraction : 2 features
* spike band power = ultra-high gamma range (250-5000 Hz) -> Characteristc of High-level cognition, perception, consciousness
* threshold crossings (tx1) = represents the moments where neural activity is considered significant -> smth is happening. (more detail: in general, neural activity is never 0, it is noisy. So we set up a threshold to decide when activity is noise or actually smth significant)

## Information in .mat files:
| Name          | Shape  | Meaning            |
|---------------|--------|--------------------|
| spikePow      | (S,)   | Gamma Band Power   |
| tx1           | (S,)   | Threshold crossing |
| sentenceText  | (S,)   | spoken sentence    |
| blockIdx      | (S,)   | block/day index    |

S = number of sentences/trials

## Hierarchy of the dataset

* files (=recording day/session)
* each file contains multiple blocks (blockIdx = 1,2,3,…)
* each block contains multiple trials (sentences)
* each trial is a matrix of shape (T_i, features) (T_i varies regarding the trial)

* -> Statistics must be computed over **blocks**

## Visual representation of a trial (preprocessed):

| Time (20 ms bins) | Area 6v spike power (128 features) | Area 6v threshold crossings (128 features) |
|------------------|-------------------------------------|--------------------------------------------|
| t = 0            | [f0, f1, ..., f127]                 | [f128, f129, ..., f255]                    |
| t = 1            | [f0, f1, ..., f127]                 | [f128, f129, ..., f255]                    |
| t = 2            | [f0, f1, ..., f127]                 | [f128, f129, ..., f255]                    |
| ...              | ...                                 | ...                                        |
| t = T_i − 1       | [f0, f1, ..., f127]                 | [f128, f129, ..., f255]                    |


Each row describes a 20ms duration
T_i = number of rows