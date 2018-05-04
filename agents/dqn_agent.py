import torch.nn as nn
import torch.optim as optim
from utils.replay_buffer import ReplayBuffer
from collections import namedtuple


class DQNAgent:

    def __init__(self, model_class, model_params, rng, device='cpu', n_step=1, n_episodes=1000, mb_size=32,
                 lr=8e-3, momentum=0.9, criterion=nn.SmoothL1Loss, optimizer=optim.RMSprop, gamma=0.99,
                 epsilon_scheduler=None, epsilon_scheduler_use_steps=True, target_update_frequency=1e4,
                 parameter_update_frequency=1, replay_buffer_size=100000, replay_build_wait=None,
                 replay_build_wait_use_steps=True):

        self.model_class = model_class
        self.rng = rng
        self.device = device
        self.n_step = n_step
        self.n_episodes = n_episodes
        self.mb_size = mb_size
        self.lr = lr
        self.momentum = momentum
        self.criterion = criterion
        self.optimizer = optimizer
        self.gamma = gamma
        self.epsilon_scheduler = epsilon_scheduler
        self.epsilon_scheduler_use_steps = epsilon_scheduler_use_steps
        self.model_learner = self.model_class(*model_params)
        self.model_target = self.model_class(*model_params)
        self.target_update_frequency = target_update_frequency
        self.parameter_update_frequency = parameter_update_frequency
        self.replay_buffer_size = replay_buffer_size
        if self.replay_buffer_size > 0:
            if replay_build_wait:
                assert replay_build_wait <= self.replay_buffer_size
                self.replay_build_wait = replay_build_wait
            else:
                self.replay_build_wait = self.mb_size
            self.replay_buffer = ReplayBuffer(self.replay_buffer_size)
            self.replay_build_wait_use_steps = replay_build_wait_use_steps
        self.model_target.load_state_dict(self.model_learner.state_dict())
        self.elapsed_steps = 0
        self.elapsed_episodes = 0

    def step(self, env):

    def step_fn(self, env):
        Transition = namedtuple('Transition', 'state action reward next_state done')
        returns = []
        now = time.time()
        for ep in range(self.n_episodes):
            o = env.reset()
            done = False
            ret = 0
            while not done:
                step += 1
                if step <= replay_start_size:
                    action = np.random.randint(0, env.action_space.n)
                else:
                    if np.random.random() <= eps:
                        action = np.random.randint(0, env.action_space.n)
                    else:
                        model_learner.eval()
                        action = model_learner(torch.tensor(o, device=device, dtype=torch.float).unsqueeze(0)).max(1)[
                            1].detach().to('cpu').numpy()[0]
                o_, rew, done, info = env.step(action)
                # add to replay buffer
                buffer.insert(Transition(o, action, rew, o_, done))
                if step <= replay_start_size:
                    continue
                if done and ret < 200:
                    rew = rew
                ret += rew
                '''if (update_ctr+1) % param_update_freq != 0:
                    update_ctr += 1
                    continue'''
                samples = buffer.sample(minibatch_size)
                batch = Transition(*zip(*samples))
                non_final_mask = torch.tensor(tuple(map(lambda s: not s, batch.done)), device=device)
                states = torch.tensor(batch.state, device=device, dtype=torch.float)
                actions = torch.tensor(batch.action, device=device, dtype=torch.long)
                rewards = torch.tensor(batch.reward, device=device, dtype=torch.float)
                non_final_states = torch.tensor([s for s, d in zip(batch.next_state, batch.done) if not d],
                                                device=device, dtype=torch.float)
                # model_learner.train()
                outputs = model_learner(states).gather(1, actions.view(actions.size()[0], -1))
                if done:
                    targets = rewards
                else:
                    targets = torch.zeros(minibatch_size, device=device)
                    # model_learner.eval()
                    targets[non_final_mask] += model_target(non_final_states).max(1)[0].detach()
                    targets += rewards
                    # model_learner.train()

                loss = criterion(outputs, targets.view(targets.size()[0], -1))
                optimizer.zero_grad()
                loss.backward()
                o = o_
                for p in model_learner.parameters():
                    p.grad.data.clamp(-1, 1)
                optimizer.step()
                if step % targ_update_freq == 0:
                    model_target.load_state_dict(model_learner.state_dict())
            returns.append(ret)
            # if eps > 0.01:
            eps *= eps_red
            lr = max(lr * .98, 125e-7)
            if ep % 100 == 0 and len(returns) >= 100:
                print('mean prev 100 returns:', ep, ':', np.mean(returns[-100:]))
        print('finished in:', time.time() - now)