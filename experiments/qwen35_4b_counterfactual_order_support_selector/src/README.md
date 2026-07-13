# Source

`selector.py` contains the label-free counterfactual prediction rules, the
explicit oracle-balanced mismatch control, paired task bootstrap, and frozen
gate evaluation. Prediction APIs are separated from grading so unit tests can
prove that hidden-label mutation cannot change deployable outputs.
