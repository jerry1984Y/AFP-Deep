# AFP-Deep
Improving Antifreeze Proteins Prediction with Protein Language Models and Hybrid Feature Extraction Network

# 1. Citation
coming soon

# 2. Requirements

Python >= 3.10.6

torch = 2.0.0

pandas = 2.0.0

scikit-learn = 1.2.2

ncbi-blast = 2.2.26

ProtTrans (ProtT5-XL-UniRef50 model)

# 3. Description
The proposed AFP-Deep method is implemented using Python on torch. The model learns sequence global contextual features  using pre-trained language models (pLMs) using transfer learning and the hybrid deep neural network designed for evolutionary context extraction enhances the correlation between evolutionary embedding and antifreeze pattern.


# 4 DatasetS
## 4.1 Dateset481
The dataset is obtained from Kandaswamy et. al [1], containing 481 antifreeze proteins and 9493 non-antifreeze proteins. The Files in Dateset dictionary are:

AFP.fasta:this file contains 481 AFPs with fasta format

non-AFP.fasta: this file contains 9493 no-AFPs with fasta format

afp.seq: this file contains 481 AFPs with key-value format and the data from AFP.fasta

non-afp.seq: this file contains 9493 no-AFPs with key-value format and the data from non-AFP.fasta

orderafp: this file contains only AFPs names from the AFP.fasta for facility using

ordernon_afp9493:this file contains only no-AFPs names from the non-AFP.fasta for facility using.

## 4.2 Dateset920
The dateset is constructed in our work, containing 920 antifreeze proteins and 3948 non-antifreeze proteins.The Files in Dateset dictionary are:

AFP920.fasta:this file contains 920 AFPs with fasta format

non-AFP3948.fasta: this file contains 3948 no-AFPs with fasta format

afp920.seq: this file contains 920 AFPs with key-value format and the data from AFP.fasta

non-afp3948.seq: this file contains 3948 no-AFPs with key-value format and the data from non-AFP.fasta

orderafp920: this file contains only AFPs names from the AFP.fasta for facility using

ordernon_afp3948:this file contains only no-AFPs names from the non-AFP.fasta for facility using.

# 5. How to Use

## 5.1 Set up environment for PSSM and ProtTrans
1. Download ncbi-blast and related dataset follow procedure from https://blast.ncbi.nlm.nih.gov/doc/blast-help/downloadblastdata.html 
2. Set ProtTrans follow procedure from https://github.com/agemagician/ProtTrans/tree/master.

## 5.2 Extract features

1. Extract PSSM feature: cd to the AFP-TLDeep dictionary,and run "python3 AFP_PSSM_blastpgp_calc.py",the PSSM matrixs will be extracted to midData/PSSM fold, and then run "pythons3 AFP_PSSM_ORI_20_calc.py" to extract the pssm features in midData/PSSM_ORI_20.

2. Extract pLMs embedding: cd to the AFP-TLDeep dictionary, and run "python3 pLMs_Extraction.py", the pLMs embedding matrixs will be extracted to midData/ProtTrans fold.

## 5.3 Train and test

1. Cd to the AFP-TLDeep dictionary,and run "python3 AFP-TLDeep.py"

4. The comparison work in our paper refers "ProtT5-BiLSTM.py", "ProtT5-MLP.py", "PSSM_CNN_BiLSTM_Liner.py","PSSM_CNN_Self_Attention_Liner.py","PSSM_ResNet_BiLSTM_Liner.py","PSSM_ResNet_SelfAttention_Liner.py","Seq_One_HOT.py","Seq_Word_Embedding.py"

# References
[1] Kandaswamy, Krishna Kumar, et al. "AFP-Pred: A random forest approach for predicting antifreeze proteins from sequence-derived properties." Journal of theoretical biology 270.1 (2011): 56-62.
