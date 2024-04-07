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
Accurate identification of antifreeze proteins (AFPs) is crucial in developing biomimetic synthetic anti-icing materials and low-temperature organ preservation materials. Although numerous machine-learning methods have been proposed for AFPs prediction, the complex and diverse nature of AFPs limits the prediction performance of existing methods. In this study, we propose AFP-Deep, a new deep learning method to predict antifreeze proteins by integrating deep embedding from protein sequences with pre-trained protein language models and evolutionary contexts with hybrid feature extraction networks. The experimental results demonstrated that the main advantage of AFP-Deep is its utilization of pre-trained protein language models, which can extract discriminative global contextual features from protein sequences. Additionally, the hybrid deep neural networks designed for protein language models and evolutionary context feature extraction enhances the correlation between embeddings and antifreeze pattern. The performance evaluation results show that AFP-Deep achieves superior performance compared to state-of-the-art models on benchmark datasets, achieving an AUPRC of 0.724 and 0.924, respectively.


# 4 Datasets
## 4.1 Dateset481
The dataset is obtained from Kandaswamy et. al [1], containing 481 antifreeze proteins and 9493 non-antifreeze proteins. The Files in Dateset dictionary are:

AFP481.seq: this file contains 481 AFPs with key-value format

Non-AFP9493.seq: this file contains 9493 Non-AFPs with key-value format

## 4.2 Dateset920
The dateset is constructed in our work, containing 920 antifreeze proteins and 3955 non-antifreeze proteins.The Files in Dateset dictionary are:


AFP920.seq: this file contains 920 AFPs with key-value format

Non-AFP3955.seq: this file contains 3955 Non-AFPs with key-value format


# 5. How to Use

## 5.1 Set up environment for PSSM and ProtTrans
1. Download ncbi-blast and related dataset follow procedure from https://blast.ncbi.nlm.nih.gov/doc/blast-help/downloadblastdata.html 
2. Set ProtTrans follow procedure from https://github.com/agemagician/ProtTrans/tree/master.

## 5.2 Extract features

1. Extract PSSM feature: cd to the AFP-Deep dictionary,and run "python3 AFP_Pssm_calc.py",the PSSM matrixs will be extracted to midData/PSSM fold, and then run "pythons3 AFP_PSSM_ORI_20_calc.py" to extract the pssm features in midData/PSSM_ORI_20. Prior to running AFP_Pssm_calc.py, please modify the "dir_pssm" and "ncbi-blast-2.2.26/bin" paths in the code.

2. Extract pLMs embedding: cd to the AFP-Deep dictionary, and run "python3 pLMs_Extraction.py", the pLMs embedding matrixs will be extracted to midData/ProtTran fold.

## 5.3 Training and testing

1. Cd to the AFP-Deep dictionary,and run "python3 AFP-Deep.py"

4. The comparison work in our paper refers "ProtT5-BiLSTM.py", "ProtT5-MLP.py", "PSSM_CNN_BiLSTM_Liner.py", "PSSM_CNN_Self_Attention_Liner.py", "PSSM_ResNet_BiLSTM_Liner.py", "PSSM_ResNet_SelfAttention_Liner.py", "Seq_One_HOT.py", "Seq_Word_Embedding.py"

# References
[1] Kandaswamy, Krishna Kumar, et al. "AFP-Pred: A random forest approach for predicting antifreeze proteins from sequence-derived properties." Journal of theoretical biology 270.1 (2011): 56-62.
