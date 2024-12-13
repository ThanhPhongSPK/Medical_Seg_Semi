import torch 

def save_net_opt(net, optimizer, path):
    state = {
        'net': net.state_dict(),
        'optim': optimizer.state_dict()
    }
    torch.save(state, str(path))

def load_net(net, path):
    state = torch.load(str(path))
    net.load_state_dict(state['net']) 

def load_net_opt(net, optimizer, path): 
    state = torch.load(str(path))
    net.load_state_dict(state['net'])
    optimizer.load_state_dict(state['optim'])