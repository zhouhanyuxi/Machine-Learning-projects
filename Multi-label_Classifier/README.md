# Multi-label classifier
2020.2.20 - 2020.3.8

### Introduction to Multi-label classification
In contrast to multi-class classification in which instances can only belong to a single class, in multi-label classification1 problems instances can belong to more than one class at a time. A selection of multi-label classification approaches based on ensemble methods exist in the literature.  

(1) Binary Relevance algorithm 
Individual binary base classifiers are implemented for each label. This uses a one-vs-all approach to generate the training sets for each base classifier.  
The training datasets for base classifiers can be very imbalanced because of the one-vs-all approach. So under-sampling approach is needed.  

(2) Classifier Chains
The classifier chains algorithm is an effective multi-label classification algorithm that takes advantage of label associations.  
A classifier chain model generates a chain of binary classifiers each of predicts the presence or absence of a specific label. The input to each classifier in the chain, however, includes the original descriptive features plus the outputs of the classifiers so far in the chain. This allows label associations to be taken into account.  

### My work
Write my own implementation the binary relevance and classifier chain algorithms using Numpy with scikit-learn implementations for the base estimators.  
Implement the under-sampling approach myself.  

I used only two different base estimators in this assignment. Logistic Regression usually runs faster than Decision Tree, the default one and with undersampling, it preforms much better than decition tree. Decision tree without limitation of depth turns to be overfitted easily and takes more time generally, and under sampling doesn't improve it much as well.

As for the differences between binary relevance and classifier chains, the previous one does fewer calculations than the latter and under sampling helps to generalize more when it is used with binary relevance. So binary relavance can generalize well with really few calculations.
