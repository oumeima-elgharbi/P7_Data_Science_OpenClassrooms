import uvicorn
import streamlit

import warnings
from time import time, strftime, gmtime

import os
from os import listdir
from os.path import isfile, join

import unittest
import pytest
import virtualenv
import pydantic

import sys
import shap
import sklearn
import streamlit

print("User Current Version:-", sys.version)
print("Scikit-Learn Version : {}".format(sklearn.__version__))
print("SHAP Version : {}".format(shap.__version__))
print("Streamlit Version : {}".format(streamlit.__version__))
