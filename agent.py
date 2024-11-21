from network import DDQN, Encoder, Decoder
import copy
import torch
import torch.optim as optim
import torch.distributions as dist
import torch.nn.functional as F

class OfflineAHTAgent(object):
    def __init__(self, state_size, action_size, human_action_size, layer_size, lstm_dim, encoding_dim, device="cpu"):
        self.state_size = state_size
        self.action_size = action_size
        self.human_action_size = human_action_size
        self.layer_size = layer_size
        self.lstm_dim = lstm_dim
        self.encoding_dim = encoding_dim
        self.device = device
        self.gamma = 0.99
        self.total_updates = 0

        self.encoder = Encoder(state_size+human_action_size, lstm_dim, encoding_dim).double().to(self.device)
        self.decoder = Decoder(encoding_dim, layer_size, human_action_size).double().to(self.device)

        self.value_network = DDQN(state_size+human_action_size+encoding_dim, action_size, layer_size).double().to(self.device)
        self.target_value_network = copy.deepcopy(self.value_network).double().to(self.device)

        self.optimizer = optim.Adam([*self.encoder.parameters(), *self.decoder.parameters(), *self.value_network.parameters()], lr=1e-4)
        self.lstm_hiddens_train = None
        self.lstm_hiddens_eval = None

    def act(self, input_obs):
        
        input_tensor = torch.tensor(input_obs).double().to(self.device)
        if self.lstm_hiddens_eval is None:
            self.lstm_hiddens_eval = (torch.zeros(1, input_tensor.size()[0], self.lstm_dim).double().to(self.device), torch.zeros(1, input_tensor.size()[0], self.lstm_dim).double().to(self.device))
        rep, self.lstm_hiddens_eval = self.encoder(input_tensor, self.lstm_hiddens_eval)
        action_vals = self.value_network(torch.cat([input_tensor]))

        return torch.argmax(action_vals, dim=-1)

    def train(self, input_obs, human_actions, ai_actions, input_dones, input_rews, input_nobs):
        input_tensor = torch.tensor(input_obs).double().to(self.device)
        human_actions_tensor = torch.tensor(human_actions).double().to(self.device)
        ai_actions_tensor = torch.tensor(ai_actions).double().to(self.device)
        dones_tensor = torch.tensor(input_dones).double().to(self.device)
        rews_tensor = torch.tensor(input_rews).double().to(self.device)
        nobs_tensor = torch.tensor(input_nobs).double().to(self.device)

        batch_size = input_tensor.size()[0]
        seq_length = input_tensor.size()[1]

        self.lstm_hiddens_train = (torch.zeros(1, input_tensor.size()[0], self.lstm_dim).double().to(self.device), torch.zeros(1, input_tensor.size()[0], self.lstm_dim).double().to(self.device))
        predicted_action_logits = []
        predicted_action_vals = []
        target_action_vals = []
        for t in range(seq_length):
            obs_t = input_tensor[:, t, :]
            nobs_t = nobs_tensor[:, t, :]
            rep, self.lstm_hiddens_eval = self.encoder(obs_t, self.lstm_hiddens_train)
            decoded_action_logits = self.decoder(rep)
            predicted_action_logits.append(decoded_action_logits.unsqueeze(1))
            action_vals = self.value_network(torch.cat([obs_t, rep.detach()], dim=-1))
            predicted_action_vals.append(action_vals.unsqueeze(1))

            updated_rep, _ = self.encoder(nobs_t, self.lstm_hiddens_train)
            target_action_val = self.target_value_network(torch.cat([nobs_t, updated_rep], dim=-1))
            target_action_vals.append(target_action_val.unsqueeze(1))

        all_predicted_logits = torch.cat(predicted_action_logits, dim=1)
        action_dist = dist.OneHotCategorical(logits=all_predicted_logits)
        action_log_probs = action_dist.log_prob(human_actions_tensor)

        enc_dec_loss = -action_log_probs.mean()

        selected_action = ai_actions_tensor.argmax(dim=-1).unsqueeze(dim=-1)
        all_predicted_vals = torch.cat(predicted_action_vals, dim=1).gather(-1, selected_action)
        all_target_vals, _ = torch.cat(target_action_vals, dim=1).max(dim=-1)

        target_vals = rews_tensor.unsqueeze(-1) + self.gamma * (1-dones_tensor.unsqueeze(-1)) * all_target_vals.unsqueeze(-1)
        usual_q_loss = ((all_predicted_vals - target_vals.detach())**2).mean()

        cql_prob_dist =  dist.OneHotCategorical(logits=torch.cat(predicted_action_vals, dim=1))
        cql_log_probs_loss = -cql_prob_dist.log_prob(ai_actions_tensor).mean()

        total_loss = enc_dec_loss+usual_q_loss+cql_log_probs_loss
        print("Losses : ",enc_dec_loss, usual_q_loss, cql_log_probs_loss)

        
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        self.total_updates += 1
        if self.total_updates % 100 == 0:
            self.target_value_network = copy.deepcopy(self.value_network).double().to(self.device)
            

class OfflineAHTAgentV2(object):
    def __init__(self, state_size, action_size, human_action_size, layer_size, lstm_dim, encoding_dim, device="cpu"):
        self.state_size = state_size
        self.action_size = action_size
        self.human_action_size = human_action_size
        self.layer_size = layer_size
        self.lstm_dim = lstm_dim
        self.encoding_dim = encoding_dim
        self.device = device
        self.gamma = 0.99
        self.total_updates = 0

        self.encoder = Encoder(state_size+human_action_size, lstm_dim, encoding_dim).double().to(self.device)
        self.decoder = Decoder(encoding_dim, layer_size, human_action_size).double().to(self.device)

        self.value_network = DDQN(state_size+human_action_size+encoding_dim, action_size*human_action_size, layer_size).double().to(self.device)
        self.target_value_network = copy.deepcopy(self.value_network).double().to(self.device)

        self.optimizer = optim.Adam([*self.encoder.parameters(), *self.decoder.parameters(), *self.value_network.parameters()], lr=1e-4)
        self.lstm_hiddens_train = None
        self.lstm_hiddens_eval = None

    def act(self, input_obs):
        
        input_tensor = torch.tensor(input_obs).double().to(self.device)
        if self.lstm_hiddens_eval is None:
            self.lstm_hiddens_eval = (torch.zeros(1, input_tensor.size()[0], self.lstm_dim).double().to(self.device), torch.zeros(1, input_tensor.size()[0], self.lstm_dim).double().to(self.device))
        rep, self.lstm_hiddens_eval = self.encoder(input_tensor, self.lstm_hiddens_eval)
        action_vals = self.value_network(torch.cat([input_tensor, rep.detach()], dim=-1))

        reshaped_action_vals = action_vals.view(-1, self.action_size, self.human_action_size)
        human_action_logits = self.decoder(rep.detach()).unsqueeze(-1).repeat(1, self.action_size, 1)
        human_action_probs = F.softmax(human_action_logits, dim=-1)

        # aggregated q-vals
        aggregated_q_val = (reshaped_action_vals*human_action_probs).sum(dim=-1)
        return torch.argmax(aggregated_q_val, dim=-1)

    def train(self, input_obs, human_actions, ai_actions, input_dones, input_rews, input_nobs):
        input_tensor = torch.tensor(input_obs).double().to(self.device)
        human_actions_tensor = torch.tensor(human_actions).double().to(self.device)
        ai_actions_tensor = torch.tensor(ai_actions).double().to(self.device)
        dones_tensor = torch.tensor(input_dones).double().to(self.device)
        rews_tensor = torch.tensor(input_rews).double().to(self.device)
        nobs_tensor = torch.tensor(input_nobs).double().to(self.device)

        batch_size = input_tensor.size()[0]
        seq_length = input_tensor.size()[1]

        self.lstm_hiddens_train = (torch.zeros(1, input_tensor.size()[0], self.lstm_dim).double().to(self.device), torch.zeros(1, input_tensor.size()[0], self.lstm_dim).double().to(self.device))
        predicted_action_logits = []
        predicted_action_vals = []
        target_action_vals = []
        act_probs = []

        for t in range(seq_length):
            obs_t = input_tensor[:, t, :]
            nobs_t = nobs_tensor[:, t, :]
            rep, self.lstm_hiddens_eval = self.encoder(obs_t, self.lstm_hiddens_train)
            decoded_action_logits = self.decoder(rep)

            predicted_action_logits.append(decoded_action_logits.unsqueeze(1))
            action_vals = self.value_network(torch.cat([obs_t, rep.detach()], dim=-1))
            predicted_action_vals.append(action_vals.unsqueeze(1))

            updated_rep, _ = self.encoder(nobs_t, self.lstm_hiddens_train)
            joint_target_action_val = self.target_value_network(torch.cat([nobs_t, updated_rep], dim=-1))
            n_state_logits = self.decoder(updated_rep.detach()).unsqueeze(1).repeat(1, self.action_size, 1)
            human_action_probs = F.softmax(n_state_logits, dim=-1)

            target_action_val = (joint_target_action_val.view(-1, self.action_size, self.human_action_size) * human_action_probs).sum(dim=-1)
            target_action_vals.append(target_action_val.unsqueeze(1))

        # Compute encoder-decoder loss
        all_predicted_logits = torch.cat(predicted_action_logits, dim=1)
        action_dist = dist.OneHotCategorical(logits=all_predicted_logits)
        action_log_probs = action_dist.log_prob(human_actions_tensor)
        enc_dec_loss = -action_log_probs.mean()

        # Train value network
        ai_selected_action = ai_actions_tensor.argmax(dim=-1).unsqueeze(dim=-1)
        human_selected_action = human_actions_tensor.argmax(dim=-1).unsqueeze(dim=-1)
        selected_action = ai_selected_action*self.human_action_size + human_selected_action

        all_predicted_vals = torch.cat(predicted_action_vals, dim=1).gather(-1, selected_action)
        all_target_vals, _ = torch.cat(target_action_vals, dim=1).max(dim=-1)

        target_vals = rews_tensor.unsqueeze(-1) + self.gamma * (1-dones_tensor.unsqueeze(-1)) * all_target_vals.unsqueeze(-1)
        usual_q_loss = ((all_predicted_vals - target_vals.detach())**2).mean()

        # Add CQL regularizer
        reshaped_q_vals = torch.cat(predicted_action_vals, dim=1).view(
            -1, seq_length, self.action_size, self.human_action_size
        )
        repeated_predicted_probs = F.softmax(all_predicted_logits.unsqueeze(2).repeat(1, 1, self.action_size, 1)).detach()
        all_q_vals = (reshaped_q_vals*repeated_predicted_probs).sum(dim=-1)
        
        cql_prob_dist =  dist.OneHotCategorical(logits=all_q_vals)
        cql_log_probs_loss = -cql_prob_dist.log_prob(ai_actions_tensor).mean()

        total_loss = enc_dec_loss+usual_q_loss+cql_log_probs_loss
        print("Losses : ",enc_dec_loss, usual_q_loss, cql_log_probs_loss)

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        self.total_updates += 1
        if self.total_updates % 100 == 0:
            self.target_value_network = copy.deepcopy(self.value_network).double().to(self.device)
            

