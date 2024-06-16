from mindspore import nn
from mindspore.common.parameter import Parameter
from mindspore.common.initializer import initializer, Tensor
import mindspore.common.dtype as mstype
from mindspore._extends import cell_attr_register
from mindspore.nn.cell import Cell
from mindspore.ops import functional as F
from mindspore.ops import operations as P

class Dropout(nn.Cell):
    r"""
        A Dropout Implements with P.Dropout and  P.DropoutDoMask for parallel training.
    """

    def __init__(self, keep_prob=0.5, dtype=mstype.float32):
        super(Dropout, self).__init__()
        if keep_prob <= 0 or keep_prob > 1:
            raise ValueError(
                "dropout probability should be a number in range (0, 1], but got {}".format(
                    keep_prob))
        self.keep_prob = keep_prob
        self.dropout = P.Dropout(keep_prob)

    def construct(self, x):
        r"""
           Input: a tensor
           Returns: a tensor
        """
        if not self.training:
            return x

        if self.keep_prob == 1:
            return x

        out, _ = self.dropout(x)
        return out