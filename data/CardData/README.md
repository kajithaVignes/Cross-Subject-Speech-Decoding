# Data for: An accurate and rapidly calibrating speech neuroprosthesis

Dryad DOI: https://doi.org/10.5061/dryad.dncjsxm85

Nicholas S. Card, Maitreyee Wairagkar, Carrina Iacobacci, Xianda Hou, Tyler Singer-Clark, Francis R. Willett, Erin M. Kunz, Chaofei Fan, Maryam Vahdati Nia, Darrel R. Deo, Aparna Srinivasan, Eun Young Choi, Matthew F. Glasser, Leigh R. Hochberg, Jaimie M. Henderson, Kiarash Shahlaie, Sergey D. Stavisky*, and David M. Brandman*.

* "*" denotes co-senior authors

## Overview

This repository contains the data necessary to reproduce the results of the paper "*An Accurate and Rapidly Calibrating Speech Neuroprosthesis*" by Card et al. (2024), *N Eng J Med*.

The code is written primarily in Python and is hosted on [GitHub](https://github.com/Neuroprosthetics-Lab/nejm-brain-to-text).

The data can be downloaded from this Dryad repository. Please download this data and place it in the `data` directory of the GitHub code. All included data has been anonymized and does not include any identifiable information.

## Files:

* `t15_copyTask.pkl`
  * Data from Copy Task trials during evaluation blocks (1,718 total trials) are necessary for reproducing the online decoding performance plots (Figure 2).
  * Copy Task data includes, for each trial: cue sentence, decoded phonemes and words, trial duration, and RNN-predicted logits.
* `t15_personalUse.pkl`
  * Data from Conversation Mode (22,126 total sentences) is necessary for reproducing Figure 4.
  * Conversation Mode data includes, for each trial, the number of decoded words, the sentence duration, and the participant's rating of how correct the decoded sentence was.
  * Specific decoded sentences from Conversation Mode are not included to protect the participant's privacy.
* `t15_copyTask_neuralData.zip`
  * Processed neural data and sentence labels during 11,000+ Copy Task trials, from 45 data collection sessions spanning 20 months.
  * Processed neural data is threshold crossings (-4.5 RMS threshold) and spike band power for each of the 256 recording channels (512 features total) at 20 ms resolution. Data are normalized (z-scored) based on the preceding 20 trials.
  * Trials are split into "train", "val", and "test" trials. "test" trials do not include the ground truth sentence label as they will be used for the Brain-to-Text 2025 challenge.
  * Refer to the [GitHub repo](https://github.com/Neuroprosthetics-Lab/nejm-brain-to-text) for specific instructions on how to load and use this data, and refer to the Brain-to-Text '25 competition page for details on how to submit your own results.
* `t15_pretrained_rnn_baseline.zip`
  * An RNN model that has been pretrained on the T15 copyTask neural dataset.
  * Refer to the [GitHub repo](https://github.com/Neuroprosthetics-Lab/nejm-brain-to-text) for specific instructions on how to load and do inference with this model.

## Change log

**3 July 2025:** Added two zip files. One is fully de-identified processed neural data. The other is the weights for a pretrained RNN model. No personal or identifiable information is included in any of the data included here. 
* t15_copyTask_neuralData.zip
* t15_pretrained_rnn_baseline.zip