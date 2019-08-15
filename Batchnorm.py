class BatchNorm(nn.Module):
    def __init__(self, num_features):
        super(BatchNorm, self).__init__()
        self.num_features = num_features
        self.gamma = nn.Parameter(torch.randn(self.num_features))
        self.beta = nn.Parameter(torch.randn(self.num_features))
        self.moving_mean = Variable(torch.zeros(self.num_features))
        self.moving_var = Variable(torch.zeros(self.num_features))
        self.is_training=True
        self.eps=1e-5
        self.momentum=0.1

    def forward(self, X):
        outputs, self.moving_mean, self.moving_var = batch_norm(X, self.gamma, self.beta,self.moving_mean, self.moving_var,
            self.is_training, self.eps, self.momentum)
        return outputs


def batch_norm(X, gamma, beta, moving_mean, moving_var, is_training=True, eps=1e-5, momentum=0.1):

    if len(X.shape) == 2:#BatchNorm1d
        x_mean = torch.mean(X, dim=0, keepdim=True)
        x_var = torch.mean((X - x_mean) ** 2, dim=0, keepdim=True)
        if torch.cuda.is_available():
            x_mean=x_mean.cuda()
            x_var=x_var.cuda()
            moving_mean=moving_mean.cuda()
            moving_var=moving_var.cuda()
        if is_training:
            x_hat = (X - x_mean) / torch.sqrt(x_var + eps)
            moving_mean[:] = moving_momentum * moving_mean + (1. - moving_momentum) * x_mean
            moving_var[:] = moving_momentum * moving_var + (1. - moving_momentum) * x_var
        else:
            x_hat = (x - moving_mean) / torch.sqrt(moving_var + eps)
        outputs = gamma.view_as(x_mean) * x_hat + beta.view_as(x_mean)

    elif len(X.shape) == 4:#BatchNorm2d
        x_mean = torch.mean(X, dim=(0, 2, 3))
        x_mean = x_mean.view(1, X.size(1), 1, 1)
        x_var = torch.sqrt(torch.var(X, dim=(0, 2 , 3), unbiased=False) + eps)
        x_var = x_var.view(1, X.size(1), 1, 1)
        invstd = 1/x_var
        x_hat = (X-x_mean)*invstd
        if torch.cuda.is_available():
            x_mean=x_mean.cuda()
            x_var=x_var.cuda()
            moving_mean=moving_mean.cuda()
            moving_var=moving_var.cuda()
        if is_training:
            x_hat = (X-x_mean)*invstd
            moving_mean = momentum * moving_mean.view(1, X.size(1), 1, 1) + (1.0 - momentum) * x_mean
            moving_var = momentum * moving_var.view(1, X.size(1), 1, 1) + (1.0 - momentum) * x_var
        else:
            x_hat = (X - moving_mean.view(1, X.size(1), 1, 1)) / torch.sqrt(moving_var.view(1, X.size(1), 1, 1) + eps)

        outputs = gamma.view_as(x_mean) * x_hat + beta.view_as(x_mean)

    return outputs, moving_mean, moving_var