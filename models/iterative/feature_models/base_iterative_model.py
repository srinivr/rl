import torch.nn as nn

from models.base_model import BaseModel
from models.decoders.decoders import Decoder


class BaseIterativeModel(BaseModel):

    def __init__(self, n_input_channels, n_actions, state_embedding=128, reward_embedding=64, reward_grounding=True,
                 model_grounding=True):
        self.n_input_channels = n_input_channels
        self.n_actions = n_actions
        self.convolution_dim_out = 48  # TODO automate this!
        self.state_embedding = state_embedding
        self.reward_embedding = reward_embedding
        self.reward_grounding = reward_grounding
        self.model_grounding = model_grounding
        tuple_attributes = ['features']
        if self.reward_grounding:
            tuple_attributes.append('rewards')
        if self.model_grounding:
            tuple_attributes.append('model_td_predictions')
        super().__init__(tuple_attributes, q_values=False)
        # reward layer
        if self.reward_grounding:
            self.reward_fn = nn.Sequential(
                nn.Linear(self.state_embedding, self.reward_embedding),
                nn.ReLU(),
                nn.Linear(self.reward_embedding, self.reward_embedding),
                nn.ReLU(),
                nn.Linear(self.reward_embedding, self.n_actions)
            )
        # reconstruction layer
        if self.model_grounding:
            self.model_fn = nn.Sequential(
                nn.Linear(self.state_embedding, self.state_embedding),
                nn.ReLU(),
                nn.Linear(self.state_embedding, self.convolution_dim_out),
                nn.ReLU()
            )
            self.model_deconvolution = Decoder.get_push_decoder()

    def forward(self, x):
        x = self._get_encoding(x)
        _param_dict = {'features': x}
        if self.reward_grounding:
            _param_dict['rewards'] = self.reward_fn(x)
        if self.model_grounding:
            model = self.model_fn(x)
            model = model.view(model.size(0), 48, 1, 1)  # TODO change!
            _param_dict['model_td_predictions'] = self.model_deconvolution(model)
        return self.output_tuple(**_param_dict)

    @staticmethod
    def get_input_dimension():
        raise NotImplementedError

    def _get_encoding(self, x):
        raise NotImplementedError
