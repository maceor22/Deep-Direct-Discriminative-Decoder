# Deep-Direct-Discriminative-Decoder
# D4 paper - equation 5 best summarizes current implementation (https://arxiv.org/pdf/2205.10947.pdf)
# Need to upload rest of data (large files)

# Interesting to check out:
  # 
  # fd.py - implementation of facilitation-depression recurrent neural network
  # CosyneAbstractSubmission.pdf - short summary of work and results as of 11/19/2023
  # observation_process.py - creation and training of D4 observation process
  # state_process.py - creation and training of D4 state process
  # D4.py - implementation of D4 using particle filtering / smoothing
  # models.py - definitions for probabilistic models
  # trainers.py - definition of training scheme using maximum likelihood estimation
  # bayesian.py - implementation of Bayesian Linear layer (parameters of linear layer are treated as distributions)
  # ltc.py - implementation of Liquid Time-Constant recurrent neural network proposed by Dr. Ramin Hasani from MIT (paper link: https://arxiv.org/pdf/2006.04439.pdf)
  # ObservationModels/trained/MVN-Mix-MLP_Binomial-MLP_2023-10-14_20-2-12/P_X__Y_H/ - see here for interesting visuals showing HPD region and likelihood map in the latent space 
