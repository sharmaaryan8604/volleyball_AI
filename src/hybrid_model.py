import numpy as np

def hybrid_predict(ml_probs, markov_probs, alpha=0.8):

    final_probs = alpha * ml_probs + (1-alpha) * markov_probs

    return final_probs