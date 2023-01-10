import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore")

import sklearn
print("Scikit-Learn Version : {}".format(sklearn.__version__))

import shap
print("SHAP Version : {}".format(shap.__version__))


# SHAP : featu importance globale (constt) // local : le client 4 : telle var plus impacte sur son score et diff de feat importance
# global : 3e dans la lsite mais si client X : 1er revenu

# SMOTE : classes desequilibrées : dummy 0 : pour améliorer score sur classe 1 et réequilibrer dataset
# voir si ca améliore le score
