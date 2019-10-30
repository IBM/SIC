import torch
import torch.nn as nn


class LinRegression(nn.Linear):
    r"""Linear Regression computed in closed-form

        Args:
            X (Tensor): input tensor of size `(batch_size, num_features)`
            Y (Tensor): output tensor of size `(batch_size, num_outputs)`
            eps (float): regularization parameter
    """
    def __init__(self, X, Y, eps=1e-6):
        super().__init__(X.shape[-1], Y.shape[-1])
        # Center data
        mX = X.mean(0)
        mY = Y.mean(0)

        self.weight.data = self._lin_reg(X - mX, Y - mY, eps).t()
        with torch.no_grad():
            self.bias.data = mY - self.weight.mm(mX.view(-1, 1)).view(-1)

    def _lin_reg(self, X, Y, eps):
        CC = X.t().mm(X) / X.shape[0]
        XC = X.t().mm(Y) / X.shape[0]
        return (CC + eps * torch.eye(CC.shape[0])).inverse().mm(XC)


def linreg_reconstruct(X, idx_feature, eps=1e-6):
    """Linear regression reconstructing an input feature
        It returns a `LinRegression` object trained on reconstructing feature `idx_feature` from the others

        Args:
            X (Tensor): input tensor of size `(batch_size, num_features)`
            idx_feature (int): feature that needs to be linearly recunstructed from the rest
            eps (float): regularization parameter
    """
    idx_rest = list(set(range(X.shape[1])) - set([idx_feature]))
    return LinRegression(X[:, idx_rest], X[:, idx_feature].view(-1, 1), eps)
