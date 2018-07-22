import torch
import torch.nn as nn

from losses.auxiliary_losses.auxiliary_loss import AuxiliaryLoss


class TreeNStepRewardLoss(AuxiliaryLoss):

    def __init__(self, depth, n_step, n_proc, criterion=nn.MSELoss):
        super().__init__(criterion)
        self.depth = depth
        self.n_step = n_step
        self.n_proc = n_proc

    def get_loss(self, model_outputs, actions, rewards, batch_done, auxiliary_info):
        # Assumption: model_outputs are of stack such at 1..n_proc corresponds to t=1, next n_proc to t=2 and so on
        size = model_outputs.q_values.size()[0]
        targets = []
        outputs = []
        for i in range(size):
            curr_outputs = model_outputs.rewards
            for d in range(self.depth):
                reward_candidates = curr_outputs.rewards[i]  # 1 x size(actions)
                k = i + d * self.n_proc
                if k >= size:
                    break
                targets.append(rewards[k])
                action = actions[k]
                outputs.append(reward_candidates[action])
                if batch_done[k]:  # since depth = 0 has values always, sufficient to check here
                    break
                curr_outputs = curr_outputs.next_rewards[action]
        outputs = torch.stack(outputs, dim=0)
        targets = torch.stack(targets, dim=0)
        return self.criterion(outputs, targets)

    def get_name(self):
        return 'tree_n_step_loss'


