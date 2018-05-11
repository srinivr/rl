from collections import namedtuple
import torch.nn as nn


class BaseModel(nn.Module):
    def __init__(self, tuple_attributes=None):
        if tuple_attributes is None:
            tuple_attributes = []
        tuple_attributes.insert(0, 'q_values')
        self.output_tuple = namedtuple('Outputs', ' '.join(tuple_attributes))
        super().__init__()

    def forward(self, *input):
        """
        :param input:
        :return: namedtuple containing the attribute q_values (and optionally outputs need for auxiliary losses)
        """
        raise NotImplementedError

    def get_output_namedtuple(self):
        return self.output_tuple

    @staticmethod
    def get_input_dimension():
        raise NotImplementedError

