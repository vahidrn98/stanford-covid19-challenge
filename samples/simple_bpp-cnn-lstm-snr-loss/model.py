from common import *

# https://www.kaggle.com/xhlulu/openvaccine-simple-gru-model
# @xhlulu OpenVaccine: Simple GRU Model

class SwishFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        y = x * torch.sigmoid(x)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_variables[0]
        sigmoid = torch.sigmoid(x)
        return grad_output * (sigmoid * (1 + x * (1 - sigmoid)))
F_swish = SwishFunction.apply

class Swish(nn.Module):
    def forward(self, x):
        return F_swish(x)


class ConvBn1d(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, padding=0, dilation=1):
        super(ConvBn1d, self).__init__()
        self.conv = nn.Conv1d(in_dim, out_dim, kernel_size=kernel_size, padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm1d(out_dim)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        #x = F_swish(x)
        x = F.relu(x,True)
        x = F.dropout(x,0.2,training=self.training)
        return x



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        num_target = 5

        self.cnn = nn.Sequential(
            ConvBn1d( 16, 256, kernel_size=5, padding=2),
        )
        self.rnn = nn.Sequential(
            nn.LSTM(256, 256, num_layers=1, batch_first=True, dropout=0, bidirectional=True)
        )
        self.predict = nn.Linear(512,num_target)

    #https://discuss.pytorch.org/t/clarification-regarding-the-return-of-nn-gru/47363/2
    def forward(self, sequence):
        batch_size, length, dim = sequence.shape

        x = sequence.permute(0,2,1).contiguous()
        x = self.cnn(x)

        x = x.permute(0,2,1).contiguous()
        x, h  = self.rnn(x)
        predict = self.predict(x)
        return predict


def mse_loss(predict,target):
    batch_size,length, num_target = target.shape
    predict = predict[:,:length]
    loss = F.mse_loss(predict,target)
    return loss


# https://www.kaggle.com/c/stanford-covid-vaccine/discussion/183211
def mcrmse_loss(predict,target):
    batch_size,length, num_target = target.shape
    predict = predict[:,:length]
    predict = predict.reshape(-1,num_target)
    target  = target.reshape(-1,num_target)

    l = (predict-target)**2
    l = l.mean(0)
    l = l**0.5
    loss = l.mean()
    return loss



def snr_mcrmse_loss(predict, target, error, signal_to_noise):
    batch_size, length, num_target = target.shape
    predict = predict[:, :length]

    ###signal_to_noise = signal_to_noise+0.1 #reshape snr?
    eps = F.relu(error - 0.25)  #svm tube loss
    l = torch.abs(predict - target)
    l = F.relu(l - eps)

    l = l ** 2
    l = l * signal_to_noise.reshape(batch_size, 1, 1)
    l = l.sum(0) / signal_to_noise.sum().item()
    l = l ** 0.5
    loss = l.mean()
    return loss






##------------------------------------------------------------------
def run_check_net():
    batch_size = 8
    length = 107

    seq = np.random.choice(16,(batch_size, length))
    seq = torch.from_numpy(seq)
    seq = F.one_hot(seq,16).float()

    net = Net()
    predict = net(seq)

    print('seq',seq.shape)
    print('predict',predict.shape)


# main #################################################################
if __name__ == '__main__':
    print('%s: calling main function ... ' % os.path.basename(__file__))

    run_check_net()