"""Transition-regime trees: recursive partitioning of units by fixed attributes, with a multinomial logistic
Markov transition model (healthy, adverse, absorbing exit) in each leaf and an honest unit-level split criterion."""
from .glmtree import UnitTransferGLMTree, glm_select_depth_grouped_cv
from .multistate import MultiStateTree
__version__ = "1.0.0"
