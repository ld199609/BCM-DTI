# BCM-DTI

The identification of drug-target interaction (DTI) is significant in drug discovery and development, which is usually of high cost in time and money due to large amount of molecule and protein space. The application of deep learning in predicting DTI pairs can overcome these limitations through feature engineering. However, most works do the features extraction using the whole drug and target, which don't take the theoretical basis of pharmacological reaction that the interaction is closely related to some substructure of molecule and protein into consideration, thus poor in performance. On the other hand, some substructure-oriented studies only consider a single type of fragment, e.g., functional group.
To address these issues, we have designed a deep learning model utilizing various types of fragments to better characterize drug-target interaction.
In this work, we propose a fragment-oriented neural network for drug-target interaction and evaluated on four datasets with different distribution.

# Dataset
In the data folder, we provide all four processed datasets used in BCM-DTI: BindingDB, DAVIS, celegans and human.

# Run
You can also directly run 
  `python SSCNN_train.py`
to run the experiments.
