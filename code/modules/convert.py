import torch
import torch.nn as nn

class Convert(nn.Module):

    def __init__(self, n_in, n_out):
        super().__init__()

        self.n_in = n_in
        self.n_out = n_out

        self.linear1 = nn.Linear(n_in, n_out)
        self.linear2 = nn.Linear(n_in, n_out)

        self.reset_parameters()
    
    def __repr__(self):
        s = f"n_in={self.n_in}, n_out={self.n_out}"

        return f"{self.__class__.__name__}({s})"

    def reset_parameters(self):
        nn.init.orthogonal_(self.linear1.weight)
        nn.init.zeros_(self.linear1.bias)
        nn.init.orthogonal_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)

    def forward(self, x):
        r"""
        Args:
            x (~torch.Tensor):
                The size of each input feature is `n_in`.

        Returns:
            A tensor with the size of each output feature `n_out`.
        """
        dim = len(list(x.size()))
        if dim == 3:
            x = x.permute(2, 0, 1)
        #[n_seq, n_seq] or [n_rels, n_seq, n_seq]
        x = self.linear1(x) 
        #[n_seq, n_pos] 
        if dim == 3:
            x = x.permute(0, 2, 1)
        else:
            x = x.permute(1, 0)
        # [n_pos, n_seq]
        x = self.linear2(x)
        # [n_pos, n_pos]
        if dim == 3:
            x = x.permute(1, 2, 0)
        return x