# AFP-TLDeep
Leveraging Deep Embedding to Improve Antifreeze Proteins Prediction with Protein Language Models and Hybrid Feature Extraction Network

# Citation
coming soon

# Requirements

Python >= 3.10.6

torch = 2.0.0

pandas = 2.0.0

scikit-learn = 1.2.2

ncbi-blast = 2.2.26

Description
coming soon


# Dataset
The dataset is obtained from Kandaswamy et. al [1], containing 481 antifreeze proteins and 9493 non-antifreeze proteins. The Files in Dateset dictionary are:

AFP.fasta:this file contains 481 AFPs with fasta format

non-AFP.fasta: this file contains 9493 no-AFPs with fasta format

afp.seq: this file contains 481 AFPs with key-value format and the data from AFP.fasta

non-afp.seq: this file contains 9493 no-AFPs with key-value format and the data from non-AFP.fasta

orderafp: this file contains only AFPs names from the AFP.fasta for facility using

ordernon_afp9493:this file contais only no-AFPs names from the non-AFP.fasta for facility using.

How to Use
coming soon

References
[1] Kandaswamy, Krishna Kumar, et al. "AFP-Pred: A random forest approach for predicting antifreeze proteins from sequence-derived properties." Journal of theoretical biology 270.1 (2011): 56-62.
