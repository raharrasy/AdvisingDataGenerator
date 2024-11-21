from MarkovDataGenerator import MarkovDataGenerator
import numpy as np
from scipy.special import softmax, logsumexp
import matplotlib.pyplot as plt
from agent import OfflineAHTAgent, OfflineAHTAgentV2

if __name__ == "__main__":
    hmm = MarkovDataGenerator()
    sampled_sequences = hmm.generate_data(20000,15)
    data = hmm.remove_latent_vars(sampled_sequences)
    obs, AI_acts, human_acts, dones, rews = hmm.to_vector_form(data)
    
    ego_agent = OfflineAHTAgentV2(
        obs.shape[-1], AI_acts.shape[-1], human_acts.shape[-1], 64, 64, 32
    )

    training_iters = 50000
    batch_size = 128
    for _ in range(training_iters):
        sampled_data_ids = np.random.choice(obs.shape[0], 128, replace=False)
        ob, act, human_act, done, rew = obs[sampled_data_ids], AI_acts[sampled_data_ids], human_acts[sampled_data_ids], dones[sampled_data_ids], rews[sampled_data_ids]
        prev_acts = -np.ones_like(human_act)
        prev_acts[:, 1:, :] = human_act[:, :-1, :]
        final_ob = np.concatenate([ob, prev_acts], axis=-1)
        final_nob = -np.ones_like(final_ob)
        final_nob[:, :-1, :] = final_ob[:, 1:, :]

        ego_agent.train(final_ob, human_act, act, done, rew, final_nob)

    print("OK!!!")


