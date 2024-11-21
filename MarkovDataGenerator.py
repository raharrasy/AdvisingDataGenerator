import numpy as np
from scipy.special import softmax, logsumexp
import itertools

class MarkovDataGenerator(object):
    def __init__(self) -> None:
        super(MarkovDataGenerator, self).__init__()
        self.num_case_features = 2
        self.type_vals = ["1", "2", "3"]
        self.trust_vals = ["T", "N", "D"]

        # New outcome vals
        # self.acceptance_vals = ["A", "R", "N"]
        self.ai_advice_vals = ["X", "Y", "W"]
        self.human_answer_values = ["X", "Y"]
        self.case_data_vals = [
            "T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9", "T10", "T11", "T12", "T13", "T14", "T15"
        ]
        self.outcome_vals = ["G", "B"]

        self.type_vals_mapping = {}
        for idx, out in enumerate(self.type_vals):
            self.type_vals_mapping[out] = idx

        self.trust_vals_mapping = {}
        for idx, out in enumerate(self.trust_vals):
            self.trust_vals_mapping[out] = idx

        self.human_ans_vals_mapping = {}
        for idx, out in enumerate(self.human_answer_values):
            self.human_ans_vals_mapping[out] = idx

        self.ai_advice_vals_mapping = {}
        for idx, out in enumerate(self.ai_advice_vals):
            self.ai_advice_vals_mapping[out] = idx

        self.case_data_vals_mapping = {}
        for idx, out in enumerate(self.case_data_vals):
            self.case_data_vals_mapping[out] = idx

        self.outcome_vals_mapping = {}
        for idx, out in enumerate(self.outcome_vals):
            self.outcome_vals_mapping[out] = idx

        # Init probs for latent vars
        self.type_probs = {"1": 0.5, "2":0.3, "3": 0.2}

        # Initial attitude to AI
        self.init_trust_prior = {"1":{"T":0.8, "N": 0.1, "D":0.1}, "2":{"T":0.2, "N": 0.6, "D":0.2}, "3":{"T":0.1, "N": 0.2, "D":0.7}}
        
        # T2, T5, T10 AI agent is mostly wrong
        # t4 AI agent mostly witholds but is still right if advises
        # t11 AI agent mostly witholds but is wrong if advises
        # Rest of it mostly right
        # This could be changed for different data generation process

        self.ai_advice_probs = {
            "T1": {"X": 0.8, "Y": 0.1, "W": 0.1},
            "T2": {"X": 0.1, "Y": 0.8, "W": 0.1},
            "T3": {"X": 0.1, "Y": 0.8, "W": 0.1},
            "T4": {"X": 0.0, "Y": 0.1, "W": 0.9},
            "T5": {"X": 0.8, "Y": 0.1, "W": 0.1},
            "T6": {"X": 0.1, "Y": 0.8, "W": 0.1},
            "T7": {"X": 0.8, "Y": 0.1, "W": 0.1},
            "T8": {"X": 0.8, "Y": 0.1, "W": 0.1},
            "T9": {"X": 0.1, "Y": 0.8, "W": 0.1},
            "T10": {"X": 0.8, "Y": 0.1, "W": 0.1},
            "T11": {"X": 0.0, "Y": 0.2, "W": 0.8},
            "T12": {"X": 0.1, "Y": 0.8, "W": 0.1},
            "T13": {"X": 0.8, "Y": 0.1, "W": 0.1},
            "T14": {"X": 0.8, "Y": 0.1, "W": 0.1},
            "T15": {"X": 0.1, "Y": 0.8, "W": 0.1},
        }

        # 
        self.outcome_probs={
            "T1": {"X": {"G": 1.0, "B": 0.0}, "Y": {"G": 0.0, "B": 1.0}},
            "T2": {"X": {"G": 1.0, "B": 0.0}, "Y": {"G": 0.0, "B": 1.0}},
            "T3": {"X": {"G": 0.0, "B": 1.0}, "Y": {"G": 1.0, "B": 0.0}},
            "T4": {"X": {"G": 0.0, "B": 1.0}, "Y": {"G": 1.0, "B": 0.0}},
            "T5": {"X": {"G": 0.0, "B": 1.0}, "Y": {"G": 1.0, "B": 0.0}},
            "T6": {"X": {"G": 0.0, "B": 1.0}, "Y": {"G": 1.0, "B": 0.0}},
            "T7": {"X": {"G": 1.0, "B": 0.0}, "Y": {"G": 0.0, "B": 1.0}},
            "T8": {"X": {"G": 1.0, "B": 0.0}, "Y": {"G": 0.0, "B": 1.0}},
            "T9": {"X": {"G": 0.0, "B": 1.0}, "Y": {"G": 1.0, "B": 0.0}},
            "T10": {"X": {"G": 0.0, "B": 1.0}, "Y": {"G": 1.0, "B": 0.0}},
            "T11": {"X": {"G": 1.0, "B": 0.0}, "Y": {"G": 0.0, "B": 1.0}},
            "T12": {"X": {"G": 0.0, "B": 1.0}, "Y": {"G": 1.0, "B": 0.0}},
            "T13": {"X": {"G": 1.0, "B": 0.0}, "Y": {"G": 0.0, "B": 1.0}},
            "T14": {"X": {"G": 1.0, "B": 0.0}, "Y": {"G": 0.0, "B": 1.0}},
            "T15": {"X": {"G": 0.0, "B": 1.0}, "Y": {"G": 1.0, "B": 0.0}},
        }

        self.acceptance_probs = {
            # Agent 1 expert at T7-T15
            # Otherwise will accept advice from AI if its trustful enough
            "1":{
                "T1": {"X":{"T":{"X":0.9, "Y":0.1}, "N":{"X":0.6, "Y":0.4}, "D":{"X":0.3, "Y":0.7},}, "Y":{"T":{"X":0.1, "Y":0.9}, "N":{"X":0.4, "Y":0.6}, "D":{"X":0.7, "Y":0.3},}, "W":{"T":{"X":0.4, "Y":0.6}, "N":{"X":0.4, "Y":0.6}, "D":{"X":0.4, "Y":0.6},}},
                "T2": {"X":{"T":{"X":0.9, "Y":0.1}, "N":{"X":0.6, "Y":0.4}, "D":{"X":0.3, "Y":0.7},}, "Y":{"T":{"X":0.1, "Y":0.9}, "N":{"X":0.4, "Y":0.6}, "D":{"X":0.7, "Y":0.3},}, "W":{"T":{"X":0.4, "Y":0.6}, "N":{"X":0.4, "Y":0.6}, "D":{"X":0.4, "Y":0.6},}},
                "T3": {"X":{"T":{"X":0.9, "Y":0.1}, "N":{"X":0.6, "Y":0.4}, "D":{"X":0.3, "Y":0.7},}, "Y":{"T":{"X":0.1, "Y":0.9}, "N":{"X":0.4, "Y":0.6}, "D":{"X":0.7, "Y":0.3},}, "W":{"T":{"X":0.4, "Y":0.6}, "N":{"X":0.4, "Y":0.6}, "D":{"X":0.4, "Y":0.6},}},
                "T4": {"X":{"T":{"X":0.9, "Y":0.1}, "N":{"X":0.6, "Y":0.4}, "D":{"X":0.3, "Y":0.7},}, "Y":{"T":{"X":0.1, "Y":0.9}, "N":{"X":0.4, "Y":0.6}, "D":{"X":0.7, "Y":0.3},}, "W":{"T":{"X":0.6, "Y":0.4}, "N":{"X":0.6, "Y":0.4}, "D":{"X":0.6, "Y":0.4},}},
                "T5": {"X":{"T":{"X":0.9, "Y":0.1}, "N":{"X":0.6, "Y":0.4}, "D":{"X":0.3, "Y":0.7},}, "Y":{"T":{"X":0.1, "Y":0.9}, "N":{"X":0.4, "Y":0.6}, "D":{"X":0.7, "Y":0.3},}, "W":{"T":{"X":0.4, "Y":0.6}, "N":{"X":0.4, "Y":0.6}, "D":{"X":0.4, "Y":0.6},}},
                "T6": {"X":{"T":{"X":0.9, "Y":0.1}, "N":{"X":0.6, "Y":0.4}, "D":{"X":0.3, "Y":0.7},}, "Y":{"T":{"X":0.1, "Y":0.9}, "N":{"X":0.4, "Y":0.6}, "D":{"X":0.7, "Y":0.3},}, "W":{"T":{"X":0.6, "Y":0.4}, "N":{"X":0.6, "Y":0.4}, "D":{"X":0.6, "Y":0.4},}},
                "T7": {"X":{"T":{"X":0.9, "Y":0.1}, "N":{"X":0.9, "Y":0.1}, "D":{"X":0.9, "Y":0.1},}, "Y":{"T":{"X":0.9, "Y":0.1}, "N":{"X":0.9, "Y":0.1}, "D":{"X":0.9, "Y":0.1},}, "W":{"T":{"X":0.9, "Y":0.1}, "N":{"X":0.9, "Y":0.1}, "D":{"X":0.9, "Y":0.1},}},
                "T8": {"X":{"T":{"X":0.9, "Y":0.1}, "N":{"X":0.9, "Y":0.1}, "D":{"X":0.9, "Y":0.1},}, "Y":{"T":{"X":0.9, "Y":0.1}, "N":{"X":0.9, "Y":0.1}, "D":{"X":0.9, "Y":0.1},}, "W":{"T":{"X":0.9, "Y":0.1}, "N":{"X":0.9, "Y":0.1}, "D":{"X":0.9, "Y":0.1},}},
                "T9": {"X":{"T":{"X":0.1, "Y":0.9}, "N":{"X":0.1, "Y":0.9}, "D":{"X":0.1, "Y":0.9},}, "Y":{"T":{"X":0.1, "Y":0.9}, "N":{"X":0.1, "Y":0.9}, "D":{"X":0.1, "Y":0.9},}, "W":{"T":{"X":0.1, "Y":0.9}, "N":{"X":0.1, "Y":0.9}, "D":{"X":0.1, "Y":0.9},}},
                "T10": {"X":{"T":{"X":0.1, "Y":0.9}, "N":{"X":0.1, "Y":0.9}, "D":{"X":0.1, "Y":0.9},}, "Y":{"T":{"X":0.1, "Y":0.9}, "N":{"X":0.1, "Y":0.9}, "D":{"X":0.1, "Y":0.9},}, "W":{"T":{"X":0.1, "Y":0.9}, "N":{"X":0.1, "Y":0.9}, "D":{"X":0.1, "Y":0.9},}},
                "T11": {"X":{"T":{"X":0.9, "Y":0.1}, "N":{"X":0.9, "Y":0.1}, "D":{"X":0.9, "Y":0.1},}, "Y":{"T":{"X":0.9, "Y":0.1}, "N":{"X":0.9, "Y":0.1}, "D":{"X":0.9, "Y":0.1},}, "W":{"T":{"X":0.9, "Y":0.1}, "N":{"X":0.9, "Y":0.1}, "D":{"X":0.9, "Y":0.1},}},
                "T12": {"X":{"T":{"X":0.1, "Y":0.9}, "N":{"X":0.1, "Y":0.9}, "D":{"X":0.1, "Y":0.9},}, "Y":{"T":{"X":0.1, "Y":0.9}, "N":{"X":0.1, "Y":0.9}, "D":{"X":0.1, "Y":0.9},}, "W":{"T":{"X":0.1, "Y":0.9}, "N":{"X":0.1, "Y":0.9}, "D":{"X":0.1, "Y":0.9},}},
                "T13": {"X":{"T":{"X":0.9, "Y":0.1}, "N":{"X":0.9, "Y":0.1}, "D":{"X":0.9, "Y":0.1},}, "Y":{"T":{"X":0.9, "Y":0.1}, "N":{"X":0.9, "Y":0.1}, "D":{"X":0.9, "Y":0.1},}, "W":{"T":{"X":0.9, "Y":0.1}, "N":{"X":0.9, "Y":0.1}, "D":{"X":0.9, "Y":0.1},}},
                "T14": {"X":{"T":{"X":0.9, "Y":0.1}, "N":{"X":0.9, "Y":0.1}, "D":{"X":0.9, "Y":0.1},}, "Y":{"T":{"X":0.9, "Y":0.1}, "N":{"X":0.9, "Y":0.1}, "D":{"X":0.9, "Y":0.1},}, "W":{"T":{"X":0.9, "Y":0.1}, "N":{"X":0.9, "Y":0.1}, "D":{"X":0.9, "Y":0.1},}},
                "T15": {"X":{"T":{"X":0.1, "Y":0.9}, "N":{"X":0.1, "Y":0.9}, "D":{"X":0.1, "Y":0.9},}, "Y":{"T":{"X":0.1, "Y":0.9}, "N":{"X":0.1, "Y":0.9}, "D":{"X":0.1, "Y":0.9},}, "W":{"T":{"X":0.1, "Y":0.9}, "N":{"X":0.1, "Y":0.9}, "D":{"X":0.1, "Y":0.9},}},
            },
            # Agent 2 expert at T1-8
            # Otherwise will accept advice from AI if its trustful enough
            "2":{
                "T1": {"X":{"T":{"X":0.8, "Y": 0.2}, "N":{"X":0.8, "Y": 0.2}, "D":{"X":0.8, "Y": 0.2},}, "Y":{"T":{"X":0.8, "Y": 0.2}, "N":{"X":0.8, "Y": 0.2}, "D":{"X":0.8, "Y": 0.2},}, "W":{"T":{"X":0.8, "Y": 0.2}, "N":{"X":0.8, "Y": 0.2}, "D":{"X":0.8, "Y": 0.2},}},
                "T2": {"X":{"T":{"X":0.8, "Y": 0.2}, "N":{"X":0.8, "Y": 0.2}, "D":{"X":0.8, "Y": 0.2},}, "Y":{"T":{"X":0.8, "Y": 0.2}, "N":{"X":0.8, "Y": 0.2}, "D":{"X":0.8, "Y": 0.2},}, "W":{"T":{"X":0.8, "Y": 0.2}, "N":{"X":0.8, "Y": 0.2}, "D":{"X":0.8, "Y": 0.2},}},
                "T3": {"X":{"T":{"X":0.2, "Y": 0.8}, "N":{"X":0.2, "Y": 0.8}, "D":{"X":0.2, "Y": 0.8},}, "Y":{"T":{"X":0.2, "Y": 0.8}, "N":{"X":0.2, "Y": 0.8}, "D":{"X":0.2, "Y": 0.8},}, "W":{"T":{"X":0.2, "Y": 0.8}, "N":{"X":0.2, "Y": 0.8}, "D":{"X":0.2, "Y": 0.8},}},
                "T4": {"X":{"T":{"X":0.2, "Y": 0.8}, "N":{"X":0.2, "Y": 0.8}, "D":{"X":0.2, "Y": 0.8},}, "Y":{"T":{"X":0.2, "Y": 0.8}, "N":{"X":0.2, "Y": 0.8}, "D":{"X":0.2, "Y": 0.8},}, "W":{"T":{"X":0.2, "Y": 0.8}, "N":{"X":0.2, "Y": 0.8}, "D":{"X":0.2, "Y": 0.8},}},
                "T5": {"X":{"T":{"X":0.2, "Y": 0.8}, "N":{"X":0.2, "Y": 0.8}, "D":{"X":0.2, "Y": 0.8},}, "Y":{"T":{"X":0.2, "Y": 0.8}, "N":{"X":0.2, "Y": 0.8}, "D":{"X":0.2, "Y": 0.8},}, "W":{"T":{"X":0.2, "Y": 0.8}, "N":{"X":0.2, "Y": 0.8}, "D":{"X":0.2, "Y": 0.8},}},
                "T6": {"X":{"T":{"X":0.2, "Y": 0.8}, "N":{"X":0.2, "Y": 0.8}, "D":{"X":0.2, "Y": 0.8},}, "Y":{"T":{"X":0.2, "Y": 0.8}, "N":{"X":0.2, "Y": 0.8}, "D":{"X":0.2, "Y": 0.8},}, "W":{"T":{"X":0.2, "Y": 0.8}, "N":{"X":0.2, "Y": 0.8}, "D":{"X":0.2, "Y": 0.8},}},
                "T7": {"X":{"T":{"X":0.8, "Y": 0.2}, "N":{"X":0.8, "Y": 0.2}, "D":{"X":0.8, "Y": 0.2},}, "Y":{"T":{"X":0.8, "Y": 0.2}, "N":{"X":0.8, "Y": 0.2}, "D":{"X":0.8, "Y": 0.2},}, "W":{"T":{"X":0.8, "Y": 0.2}, "N":{"X":0.8, "Y": 0.2}, "D":{"X":0.8, "Y": 0.2},}},
                "T8": {"X":{"T":{"X":0.8, "Y": 0.2}, "N":{"X":0.8, "Y": 0.2}, "D":{"X":0.8, "Y": 0.2},}, "Y":{"T":{"X":0.8, "Y": 0.2}, "N":{"X":0.8, "Y": 0.2}, "D":{"X":0.8, "Y": 0.2},}, "W":{"T":{"X":0.8, "Y": 0.2}, "N":{"X":0.8, "Y": 0.2}, "D":{"X":0.8, "Y": 0.2},}},
                "T9": {"X":{"T":{"X":0.7, "Y": 0.3}, "N":{"X":0.5, "Y": 0.5}, "D":{"X":0.3, "Y": 0.7},}, "Y":{"T":{"X":0.3, "Y": 0.7}, "N":{"X":0.5, "Y": 0.5}, "D":{"X":0.7, "Y": 0.3},}, "W":{"T":{"X":0.55, "Y": 0.45}, "N":{"X":0.55, "Y": 0.45}, "D":{"X":0.55, "Y": 0.45},}},
                "T10": {"X":{"T":{"X":0.7, "Y": 0.3}, "N":{"X":0.5, "Y": 0.5}, "D":{"X":0.3, "Y": 0.7},}, "Y":{"T":{"X":0.3, "Y": 0.7}, "N":{"X":0.5, "Y": 0.5}, "D":{"X":0.7, "Y": 0.3},}, "W":{"T":{"X":0.45, "Y": 0.55}, "N":{"X":0.45, "Y": 0.55}, "D":{"X":0.45, "Y": 0.55},}},
                "T11": {"X":{"T":{"X":0.7, "Y": 0.3}, "N":{"X":0.5, "Y": 0.5}, "D":{"X":0.3, "Y": 0.7},}, "Y":{"T":{"X":0.3, "Y": 0.7}, "N":{"X":0.5, "Y": 0.5}, "D":{"X":0.7, "Y": 0.3},}, "W":{"T":{"X":0.55, "Y": 0.45}, "N":{"X":0.55, "Y": 0.45}, "D":{"X":0.55, "Y": 0.45},}},
                "T12": {"X":{"T":{"X":0.7, "Y": 0.3}, "N":{"X":0.5, "Y": 0.5}, "D":{"X":0.3, "Y": 0.7},}, "Y":{"T":{"X":0.3, "Y": 0.7}, "N":{"X":0.5, "Y": 0.5}, "D":{"X":0.7, "Y": 0.3},}, "W":{"T":{"X":0.55, "Y": 0.45}, "N":{"X":0.55, "Y": 0.45}, "D":{"X":0.55, "Y": 0.45},}},
                "T13": {"X":{"T":{"X":0.7, "Y": 0.3}, "N":{"X":0.5, "Y": 0.5}, "D":{"X":0.3, "Y": 0.7},}, "Y":{"T":{"X":0.3, "Y": 0.7}, "N":{"X":0.5, "Y": 0.5}, "D":{"X":0.7, "Y": 0.3},}, "W":{"T":{"X":0.45, "Y": 0.55}, "N":{"X":0.45, "Y": 0.55}, "D":{"X":0.45, "Y": 0.55},}},
                "T14": {"X":{"T":{"X":0.7, "Y": 0.3}, "N":{"X":0.5, "Y": 0.5}, "D":{"X":0.3, "Y": 0.7},}, "Y":{"T":{"X":0.3, "Y": 0.7}, "N":{"X":0.5, "Y": 0.5}, "D":{"X":0.7, "Y": 0.3},}, "W":{"T":{"X":0.45, "Y": 0.55}, "N":{"X":0.45, "Y": 0.55}, "D":{"X":0.45, "Y": 0.55},}},
                "T15": {"X":{"T":{"X":0.7, "Y": 0.3}, "N":{"X":0.5, "Y": 0.5}, "D":{"X":0.3, "Y": 0.7},}, "Y":{"T":{"X":0.3, "Y": 0.7}, "N":{"X":0.5, "Y": 0.5}, "D":{"X":0.7, "Y": 0.3},}, "W":{"T":{"X":0.55, "Y": 0.45}, "N":{"X":0.55, "Y": 0.45}, "D":{"X":0.55, "Y": 0.45},}},
            },
            # Agent 3 expert at T1-3 AND T12-15
            # Otherwise will accept advice from AI if its trustful enough
            "3":{
                "T1": {"X":{"T":{"X":0.8, "Y": 0.2}, "N":{"X":0.8, "Y": 0.2}, "D":{"X":0.8, "Y": 0.2},}, "Y":{"T":{"X":0.8, "Y": 0.2}, "N":{"X":0.8, "Y": 0.2}, "D":{"X":0.8, "Y": 0.2},}, "W":{"T":{"X":0.8, "Y": 0.2}, "N":{"X":0.8, "Y": 0.2}, "D":{"X":0.8, "Y": 0.2},}},
                "T2": {"X":{"T":{"X":0.8, "Y": 0.2}, "N":{"X":0.8, "Y": 0.2}, "D":{"X":0.8, "Y": 0.2},}, "Y":{"T":{"X":0.8, "Y": 0.2}, "N":{"X":0.8, "Y": 0.2}, "D":{"X":0.8, "Y": 0.2},}, "W":{"T":{"X":0.8, "Y": 0.2}, "N":{"X":0.8, "Y": 0.2}, "D":{"X":0.8, "Y": 0.2},}},
                "T3": {"X":{"T":{"X":0.2, "Y": 0.8}, "N":{"X":0.2, "Y": 0.8}, "D":{"X":0.2, "Y": 0.8},}, "Y":{"T":{"X":0.2, "Y": 0.8}, "N":{"X":0.2, "Y": 0.8}, "D":{"X":0.2, "Y": 0.8},}, "W":{"T":{"X":0.2, "Y": 0.8}, "N":{"X":0.2, "Y": 0.8}, "D":{"X":0.2, "Y": 0.8},}},
                "T4": {"X":{"T":{"X":0.9, "Y":0.1}, "N":{"X":0.6, "Y":0.4}, "D":{"X":0.3, "Y":0.7},}, "Y":{"T":{"X":0.1, "Y":0.9}, "N":{"X":0.4, "Y":0.6}, "D":{"X":0.7, "Y":0.3},}, "W":{"T":{"X":0.4, "Y":0.6}, "N":{"X":0.4, "Y":0.6}, "D":{"X":0.4, "Y":0.6},}},
                "T5": {"X":{"T":{"X":0.9, "Y":0.1}, "N":{"X":0.6, "Y":0.4}, "D":{"X":0.3, "Y":0.7},}, "Y":{"T":{"X":0.1, "Y":0.9}, "N":{"X":0.4, "Y":0.6}, "D":{"X":0.7, "Y":0.3},}, "W":{"T":{"X":0.6, "Y":0.4}, "N":{"X":0.6, "Y":0.4}, "D":{"X":0.6, "Y":0.4},}},
                "T6": {"X":{"T":{"X":0.9, "Y":0.1}, "N":{"X":0.6, "Y":0.4}, "D":{"X":0.3, "Y":0.7},}, "Y":{"T":{"X":0.1, "Y":0.9}, "N":{"X":0.4, "Y":0.6}, "D":{"X":0.7, "Y":0.3},}, "W":{"T":{"X":0.4, "Y":0.6}, "N":{"X":0.4, "Y":0.6}, "D":{"X":0.4, "Y":0.6},}},
                "T7": {"X":{"T":{"X":0.9, "Y":0.1}, "N":{"X":0.6, "Y":0.4}, "D":{"X":0.3, "Y":0.7},}, "Y":{"T":{"X":0.1, "Y":0.9}, "N":{"X":0.4, "Y":0.6}, "D":{"X":0.7, "Y":0.3},}, "W":{"T":{"X":0.4, "Y":0.6}, "N":{"X":0.4, "Y":0.6}, "D":{"X":0.4, "Y":0.6},}},
                "T8": {"X":{"T":{"X":0.9, "Y":0.1}, "N":{"X":0.6, "Y":0.4}, "D":{"X":0.3, "Y":0.7},}, "Y":{"T":{"X":0.1, "Y":0.9}, "N":{"X":0.4, "Y":0.6}, "D":{"X":0.7, "Y":0.3},}, "W":{"T":{"X":0.6, "Y":0.4}, "N":{"X":0.6, "Y":0.4}, "D":{"X":0.6, "Y":0.4},}},
                "T9": {"X":{"T":{"X":0.9, "Y":0.1}, "N":{"X":0.6, "Y":0.4}, "D":{"X":0.3, "Y":0.7},}, "Y":{"T":{"X":0.1, "Y":0.9}, "N":{"X":0.4, "Y":0.6}, "D":{"X":0.7, "Y":0.3},}, "W":{"T":{"X":0.6, "Y":0.4}, "N":{"X":0.6, "Y":0.4}, "D":{"X":0.6, "Y":0.4},}},
                "T10": {"X":{"T":{"X":0.9, "Y":0.1}, "N":{"X":0.6, "Y":0.4}, "D":{"X":0.3, "Y":0.7},}, "Y":{"T":{"X":0.1, "Y":0.9}, "N":{"X":0.4, "Y":0.6}, "D":{"X":0.7, "Y":0.3},}, "W":{"T":{"X":0.4, "Y":0.6}, "N":{"X":0.4, "Y":0.6}, "D":{"X":0.4, "Y":0.6},}},
                "T11": {"X":{"T":{"X":0.9, "Y":0.1}, "N":{"X":0.6, "Y":0.4}, "D":{"X":0.3, "Y":0.7},}, "Y":{"T":{"X":0.1, "Y":0.9}, "N":{"X":0.4, "Y":0.6}, "D":{"X":0.7, "Y":0.3},}, "W":{"T":{"X":0.4, "Y":0.6}, "N":{"X":0.4, "Y":0.6}, "D":{"X":0.4, "Y":0.6},}},
                "T12": {"X":{"T":{"X":0.1, "Y":0.9}, "N":{"X":0.1, "Y":0.9}, "D":{"X":0.1, "Y":0.9},}, "Y":{"T":{"X":0.1, "Y":0.9}, "N":{"X":0.1, "Y":0.9}, "D":{"X":0.1, "Y":0.9},}, "W":{"T":{"X":0.1, "Y":0.9}, "N":{"X":0.1, "Y":0.9}, "D":{"X":0.1, "Y":0.9},}},
                "T13": {"X":{"T":{"X":0.9, "Y":0.1}, "N":{"X":0.9, "Y":0.1}, "D":{"X":0.9, "Y":0.1},}, "Y":{"T":{"X":0.9, "Y":0.1}, "N":{"X":0.9, "Y":0.1}, "D":{"X":0.9, "Y":0.1},}, "W":{"T":{"X":0.9, "Y":0.1}, "N":{"X":0.9, "Y":0.1}, "D":{"X":0.9, "Y":0.1},}},
                "T14": {"X":{"T":{"X":0.9, "Y":0.1}, "N":{"X":0.9, "Y":0.1}, "D":{"X":0.9, "Y":0.1},}, "Y":{"T":{"X":0.9, "Y":0.1}, "N":{"X":0.9, "Y":0.1}, "D":{"X":0.9, "Y":0.1},}, "W":{"T":{"X":0.9, "Y":0.1}, "N":{"X":0.9, "Y":0.1}, "D":{"X":0.9, "Y":0.1},}},
                "T15": {"X":{"T":{"X":0.1, "Y":0.9}, "N":{"X":0.1, "Y":0.9}, "D":{"X":0.1, "Y":0.9},}, "Y":{"T":{"X":0.1, "Y":0.9}, "N":{"X":0.1, "Y":0.9}, "D":{"X":0.1, "Y":0.9},}, "W":{"T":{"X":0.1, "Y":0.9}, "N":{"X":0.1, "Y":0.9}, "D":{"X":0.1, "Y":0.9},}},
            },
        }

    def per_type_trust_update(self, type, ques, prev_trust, rec, ans, out):
            if type == "1":
                if ques in ["T7", "T8", "T9", "T10", "T11", "T12", "T13", "T14", "T15"]:
                    if rec == ans:
                        if out == "G":
                            if prev_trust == "T":
                                return "T"
                            elif prev_trust == "N":
                                return np.random.choice(["T", "N"], size=1, p=[0.4, 0.6])[0]
                            else:
                                return np.random.choice(["N", "D"], size=1, p=[0.4, 0.6])[0]
                        else:
                            if prev_trust == "D":
                                return "D"
                            elif prev_trust == "N":
                                return np.random.choice(["D", "N"], size=1, p=[0.4, 0.6])[0]
                            else:
                                return np.random.choice(["N", "T"], size=1, p=[0.4, 0.6])[0]
                    elif rec != ans and rec != "W":
                        if out == "G":
                            if prev_trust == "T":
                                return "N"
                            elif prev_trust == "N":
                                return "D"
                            else:
                                return "D"
                        else:
                            if prev_trust == "T":
                                return "T"
                            elif prev_trust == "N":
                                return np.random.choice(["T", "N"], size=1, p=[0.4, 0.6])[0]
                            else:
                                return np.random.choice(["N", "D"], size=1, p=[0.4, 0.6])[0]
                    elif rec == "W":
                        return prev_trust   
                else:
                    if rec == ans:
                        if out == "G":
                            if prev_trust == "T":
                                return "T"
                            elif prev_trust == "N":
                                return np.random.choice(["T", "N"], size=1, p=[0.6, 0.4])[0]
                            else:
                                return np.random.choice(["N", "D"], size=1, p=[0.6, 0.4])[0]
                        else:
                            if prev_trust == "D":
                                return "D"
                            elif prev_trust == "N":
                                return np.random.choice(["D", "N"], size=1, p=[0.6, 0.4])[0]
                            else:
                                return np.random.choice(["N", "T"], size=1, p=[0.6, 0.4])[0]
                    elif rec != ans and rec != "W":
                        if out == "G":
                            if prev_trust == "T":
                                return "N"
                            elif prev_trust == "N":
                                return "D"
                            else:
                                return "D"
                        else:
                            if prev_trust == "T":
                                return "T"
                            elif prev_trust == "N":
                                return np.random.choice(["T", "N"], size=1, p=[0.6, 0.4])[0]
                            else:
                                return np.random.choice(["N", "D"], size=1, p=[0.6, 0.4])[0]
                    elif rec == "W":
                        return prev_trust  
            elif type == "2":
                if ques in ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8"]:
                    if rec == ans:
                        if out == "G":
                            if prev_trust == "T":
                                return "T"
                            elif prev_trust == "N":
                                return np.random.choice(["T", "N"], size=1, p=[0.3, 0.7])[0]
                            else:
                                return np.random.choice(["N", "D"], size=1, p=[0.3, 0.7])[0]
                        else:
                            if prev_trust == "D":
                                return "D"
                            elif prev_trust == "N":
                                return np.random.choice(["D", "N"], size=1, p=[0.3, 0.7])[0]
                            else:
                                return np.random.choice(["N", "T"], size=1, p=[0.3, 0.7])[0]
                    elif rec != ans and rec != "W":
                        if out == "G":
                            if prev_trust == "T":
                                return "N"
                            elif prev_trust == "N":
                                return "D"
                            else:
                                return "D"
                        else:
                            if prev_trust == "T":
                                return "T"
                            elif prev_trust == "N":
                                return np.random.choice(["T", "N"], size=1, p=[0.3, 0.7])[0]
                            else:
                                return np.random.choice(["N", "D"], size=1, p=[0.3, 0.7])[0]
                    elif rec == "W":
                        return prev_trust   
                else:
                    if rec == ans:
                        if out == "G":
                            if prev_trust == "T":
                                return "T"
                            elif prev_trust == "N":
                                return np.random.choice(["T", "N"], size=1, p=[0.7, 0.3])[0]
                            else:
                                return np.random.choice(["N", "D"], size=1, p=[0.7, 0.3])[0]
                        else:
                            if prev_trust == "D":
                                return "D"
                            elif prev_trust == "N":
                                return np.random.choice(["D", "N"], size=1, p=[0.7, 0.3])[0]
                            else:
                                return np.random.choice(["N", "T"], size=1, p=[0.7, 0.3])[0]
                    elif rec != ans and rec != "W":
                        if out == "G":
                            if prev_trust == "T":
                                return "N"
                            elif prev_trust == "N":
                                return "D"
                            else:
                                return "D"
                        else:
                            if prev_trust == "T":
                                return "T"
                            elif prev_trust == "N":
                                return np.random.choice(["T", "N"], size=1, p=[0.7, 0.3])[0]
                            else:
                                return np.random.choice(["N", "D"], size=1, p=[0.7, 0.3])[0]
                    elif rec == "W":
                        return prev_trust  

            elif type == "3":
                if ques in ["T1", "T2", "T3", "T12", "T13", "T14", "T15"]:
                    if rec == ans:
                        if prev_trust == "T":
                            return "T"
                        elif prev_trust == "N":
                            return np.random.choice(["T", "N"], size=1, p=[0.3, 0.7])[0]
                        else:
                            return np.random.choice(["N", "D"], size=1, p=[0.3, 0.7])[0]
                    elif rec != ans and rec != "W":
                        if prev_trust == "D":
                            return "D"
                        elif prev_trust == "N":
                            return np.random.choice(["D", "N"], size=1, p=[0.3, 0.7])[0]
                        else:
                            return np.random.choice(["N", "T"], size=1, p=[0.3, 0.7])[0]
                    else:
                        return prev_trust   
                else:
                    if rec == ans:
                        if out == "G":
                            if prev_trust == "T":
                                return "T"
                            elif prev_trust == "N":
                                return np.random.choice(["T", "N"], size=1, p=[0.7, 0.3])[0]
                            else:
                                return np.random.choice(["N", "D"], size=1, p=[0.7, 0.3])[0]
                        else:
                            if prev_trust == "D":
                                return "D"
                            elif prev_trust == "N":
                                return np.random.choice(["D", "N"], size=1, p=[0.7, 0.3])[0]
                            else:
                                return np.random.choice(["N", "T"], size=1, p=[0.7, 0.3])[0]
                    elif rec != ans and rec != "W":
                        if out == "G":
                            if prev_trust == "T":
                                return "N"
                            elif prev_trust == "N":
                                return "D"
                            else:
                                return "D"
                        else:
                            if prev_trust == "T":
                                return "T"
                            elif prev_trust == "N":
                                return np.random.choice(["T", "N"], size=1, p=[0.7, 0.3])[0]
                            else:
                                return np.random.choice(["N", "D"], size=1, p=[0.7, 0.3])[0]
                    elif rec == "W":
                        return prev_trust  

            else:
                raise NotImplementedError

    def dict_to_mat(self, dictionary):
        levels = 1
        tested_dicts = dictionary
        while isinstance(tested_dicts[list(tested_dicts.keys())[0]], dict):
            tested_dicts = tested_dicts[list(tested_dicts.keys())[0]]
            levels += 1

        dimensions = []
        keys = []
        tested_dicts = dictionary
        for _ in range(levels):
            dimensions.append(len(list(tested_dicts.keys())))
            keys.append(list(tested_dicts.keys()))
            tested_dicts = tested_dicts[list(tested_dicts.keys())[0]]

        prob_matrix = np.zeros(tuple(dimensions))
        all_tested_keys = list(itertools.product(*keys))

        for tested_key in all_tested_keys:
            key_element_list = list(tested_key)
            set_matrix = prob_matrix
            traversed_dict = dictionary
            for idx, key in enumerate(key_element_list):
                index = None
                if key in self.type_vals_mapping.keys():
                    index = self.type_vals_mapping[key]
                elif key in self.trust_vals_mapping.keys():
                    index = self.trust_vals_mapping[key]
                elif key in self.acceptance_vals_mapping.keys():
                    index = self.acceptance_vals_mapping[key]
                elif key in self.ai_advice_vals_mapping.keys():
                    index = self.ai_advice_vals_mapping[key]
                elif key in self.case_data_vals_mapping.keys():
                    index = self.case_data_vals_mapping[key]
                else:
                    index = self.outcome_vals_mapping[key]

                traversed_dict = traversed_dict[key]
                if idx == len(key_element_list)-1:
                    set_matrix[index] = traversed_dict
                else:
                    set_matrix = set_matrix[index]

        return prob_matrix

    def generate_data(self, num_data, interaction_length):
        states = self.sample_init_states(num_data)
        sampled_data_per_step = [states]
        for id in range(interaction_length-1):
            new_states = self.sample_next_states(states, q_id=id+2)
            sampled_data_per_step.append(new_states)
            states = new_states

        return sampled_data_per_step
    
    def translate_to_continuous(self, case_data):
        continuous_inp_array = []
        for _data in case_data:
            if _data == "T1":
                continuous_inp_array.append(np.random.uniform(0, 1, 2))
            elif _data == "T2":
                continuous_inp_array.append(1+np.random.uniform(0, 1, 2))
            else:
                continuous_inp_array.append(2+np.random.uniform(0, 1, 2))

        return continuous_inp_array

    def sample_init_states(self, num_data):

        init_state = {}

        all_type_probs = [self.type_probs[type_id] for type_id in self.type_vals]
        init_types = [np.random.choice(self.type_vals, p=all_type_probs)[0] for _ in range(num_data)]
        init_state["types"] = init_types

        init_trust_val = [
            np.random.choice(self.trust_vals, p=[self.init_trust_prior[type_id][trust_val] 
            for trust_val in self.trust_vals])[0] for type_id in init_types
        ]
        init_state["trust"] = init_trust_val

        init_cases = [self.case_data_vals[0] for _ in range(num_data)]
        init_state["case"] = init_cases

        init_state_advice = [
            np.random.choice(self.ai_advice_vals, size=1, p=[self.ai_advice_probs[case_id][adv_val] 
            for adv_val in self.ai_advice_vals])[0] for case_id in init_cases
        ]
        init_state["advice"] = init_state_advice

        init_state_decision = [
            np.random.choice(self.human_answer_values, size=1, p=[self.acceptance_probs[type_id][case_id][adv_id][trust_id][acc_val] 
            for acc_val in self.human_answer_values])[0]
            for trust_id, adv_id, case_id, type_id in zip(init_trust_val, init_state_advice, init_cases, init_types)
        ]
        init_state["decision"] = init_state_decision

        init_state_outcome = [
            np.random.choice(self.outcome_vals, size=1, p=[self.outcome_probs[case_id][dec_id][adv_val] 
            for adv_val in self.outcome_vals])[0] for case_id, dec_id in zip(init_cases, init_state_decision)
        ]
        init_state["outcome_val"] = init_state_outcome
        init_state["cont_input"] = self.translate_to_continuous(init_state["case"])

        return init_state

    def sample_next_states(self, prev_states, q_id=1):
        new_state = {}
        new_state["types"] = prev_states["types"]
        new_state["case"] = ["T"+str(q_id) for _ in range(len(prev_states["types"]))]
        prev_trust = prev_states["trust"]
        prev_advice = prev_states["advice"]
        prev_outcome = prev_states["outcome_val"]
        prev_decision = prev_states["decision"] 
        #prev_case_data = prev_states["case"]

        next_trust = [
            self.per_type_trust_update(type_id, case_id, prev_tv, rec, prev_dec, out)
            for type_id, case_id, prev_tv, rec, prev_dec, out in zip(new_state["types"], prev_states["case"], prev_trust, prev_advice, prev_decision, prev_outcome)
        ]
        new_state["trust"] = next_trust

        new_state_advice = [
            np.random.choice(self.ai_advice_vals, size=1, p=[self.ai_advice_probs[case_id][adv_val] 
            for adv_val in self.ai_advice_vals])[0] for case_id in new_state["case"]
        ]
        new_state["advice"] = new_state_advice

        new_state_decision = [
            np.random.choice(self.human_answer_values, size=1, p=[self.acceptance_probs[type_id][case_id][adv_id][trust_id][acc_val] 
            for acc_val in self.human_answer_values])[0]
            for trust_id, adv_id, case_id, type_id in zip(next_trust, new_state_advice, new_state["case"], new_state["types"])
        ]
        
        new_state["decision"] = new_state_decision

        new_state_outcome = [
            np.random.choice(self.outcome_vals, size=1, p=[self.outcome_probs[case_id][dec_id][adv_val] 
            for adv_val in self.outcome_vals])[0] for case_id, dec_id in zip(new_state["case"], new_state_decision)
        ]
        new_state["outcome_val"] = new_state_outcome
        new_state["cont_input"] = self.translate_to_continuous(new_state["case"])
        return new_state


    def remove_latent_vars(self, generated_data):
        for data_dic in generated_data:
            del data_dic["trust"]
            del data_dic["types"]
            # del data_dic["case"]

        return generated_data

    def to_vector_form(self, data):
        question_one_hot_ids = np.eye(len(self.case_data_vals))
        advice_one_hot_ids = np.eye(len(self.ai_advice_vals))
        decision_one_hot_ids = np.eye(len(self.human_answer_values))
        outcome_one_hot_ids = np.eye(len(self.outcome_vals))

        question_list = []
        advice_list = []
        decision_list = []
        out_list = []
        for d_t in data:
            q_ids = [self.case_data_vals_mapping[dat_id] for dat_id in d_t["case"]]
            question_list.append(np.expand_dims(question_one_hot_ids[q_ids], axis=1))
            adv_ids = [self.ai_advice_vals_mapping[dat_id] for dat_id in d_t["advice"]]
            advice_list.append(np.expand_dims(advice_one_hot_ids[adv_ids], axis=1))
            dec_ids = [self.human_ans_vals_mapping[dat_id] for dat_id in d_t["decision"]]
            decision_list.append(np.expand_dims(decision_one_hot_ids[dec_ids], axis=1))
            out_ids = [self.outcome_vals_mapping[dat_id] for dat_id in d_t["outcome_val"]]
            out_list.append(np.expand_dims(outcome_one_hot_ids[out_ids], axis=1))

        final_q_id = np.concatenate(question_list, axis=1)
        final_adv_id = np.concatenate(advice_list, axis=1)
        final_dec_ids = np.concatenate(decision_list, axis=1)
        final_out_ids = np.concatenate(out_list, axis=1)[:, :, 0]
        final_dones = np.zeros_like(final_dec_ids)
        final_dones[:, -1, :] = 1

        return final_q_id, final_adv_id, final_dec_ids, final_dones[:, :, 0], final_out_ids

        #print(final_q_id, final_adv_id, final_dec_ids, final_out_ids)