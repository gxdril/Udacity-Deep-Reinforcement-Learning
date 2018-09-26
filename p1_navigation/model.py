import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, hidden_layers=[64,32], use_dueling=False, seed=None):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            hidden_layers (list of int): Dimensions of the hidden layers
            use_dueling (boolean): True for a dueling DQN, False for a simple DQN
            seed (int): Random seed
        """
        if len(hidden_layers) < 2:
            print("Please specify at least 2 hidden layers !!!")
            return

        super(QNetwork, self).__init__()
        if seed is not None:  self.seed = torch.manual_seed(seed)
        self.use_dueling = use_dueling
        n_layers=len(hidden_layers)
        if self.use_dueling:
            # Common network
            self.fcs = nn.ModuleList([nn.Linear(state_size,hidden_layers[0])])
            for i in range(n_layers-2):
                self.fcs.append(nn.Linear(hidden_layers[i],hidden_layers[i+1]))

            # State value network
            self.fc_val1 = nn.Linear(hidden_layers[-2],hidden_layers[-1])
            self.fc_val2 = nn.Linear(hidden_layers[-1],1)

            
            # Advantage network
            self.fc_adv1 = nn.Linear(hidden_layers[-2],hidden_layers[-1])
            self.fc_adv2 = nn.Linear(hidden_layers[-1],action_size)

        else:
            # Q network
            self.fcs = nn.ModuleList([nn.Linear(state_size,hidden_layers[0])])
            for i in range(len(hidden_layers)-1):
                self.fcs.append(nn.Linear(hidden_layers[i],hidden_layers[i+1]))
            self.fcs.append(nn.Linear(hidden_layers[-1],action_size))


    def forward(self, x):
        """Build a network that maps state -> action values."""
        
        if self.use_dueling:
            for layer in self.fcs:
                x = F.relu(layer(x))

            val = F.relu(self.fc_val1(x))
            val = self.fc_val2(val)
            
            adv = F.relu(self.fc_adv1(x))
            adv = self.fc_adv2(adv)
            q = adv + val - adv.mean()
        else:
            for layer in self.fcs[:-1]:
                x = F.relu(layer(x))
            q = self.fcs[-1](x)
        return q
