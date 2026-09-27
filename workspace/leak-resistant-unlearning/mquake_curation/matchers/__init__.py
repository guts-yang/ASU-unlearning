"""Matchers for different multi-hop reasoning types."""

from .chain_implication import match_chain_implication
from .instance_subsumption import match_instance_subsumption
from .identity_substitution import match_identity_substitution
from .multi_element_intersection import match_multi_element_intersection
from .disjunctive_elimination import match_disjunctive_elimination
from .contrapositive_reasoning import match_contrapositive_reasoning

MATCHERS = {
    "chain_implication": match_chain_implication,
    "instance_subsumption": match_instance_subsumption,
    "identity_substitution": match_identity_substitution,
    "multi_element_intersection": match_multi_element_intersection,
    "disjunctive_elimination": match_disjunctive_elimination,
    "contrapositive_reasoning": match_contrapositive_reasoning,
}
