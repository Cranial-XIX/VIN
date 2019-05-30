import copy
import numpy as np
import random
import pickle
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch.utils import data
from tqdm import tqdm

valid_room_numbers = np.array([3, 6, 9, 12])
valid_room_weights = np.array([0.1, 0.2, 0.3, 0.4])
valid_actions = np.array([[0, 2], [0, -2], [-2, 0], [2, 0]])
inverse_actions = np.array([1,0,3,2])

"""
Doors between rooms are also described as a grid cell,
therefore if the maze looks like:

Room_A - door - Room_B
  |
 door
  |
Room_C - door - Room_D

its corresponding maze looks like:

WWWWW
W   W
W WWW
W   W
WWWWW

where W means a wall. Thus, actions
are of step size 2.
"""

IMG_FOLDER = "/home/public/share/lb/images/"
DEVICE = "cuda"
BATCH_SIZE = 100
GAMMA = 0.95
MAX_N_STEPS = 20
D, W, E, G = -1, 0, 1, 2 # door, wall, emtpy, goal

def generate_maze():
    maze = np.zeros((45, 45)) # (12-1)*2*2+1
    maze.fill(W)
    n_rooms = np.random.choice(valid_room_numbers, 1, p=valid_room_weights)
    curr_n_rooms = 1
    curr_pos = np.array([22,22])
    min_x = min_y = max_x = max_y = 22
    maze[curr_pos[0], curr_pos[1]] = E

    while curr_n_rooms < n_rooms:
        action = random.choice(valid_actions)
        curr_pos += action
        x, y = curr_pos
        if maze[x,y] == W:
            curr_n_rooms += 1
            maze[x,y] = E
            maze[x-action[0]//2, y-action[1]//2] = D # draw the door
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            max_y = max(max_y, y)
    center_x = (max_x+min_x)//2
    center_y = (max_y+min_y)//2
    return maze[center_x-11:center_x+12,center_y-11:center_y+12]

def generate_data(n_maps=20000):
    mazes = []
    for _ in tqdm(range(n_maps)):
        maze = generate_maze()
        mazes.append(maze)

    split = int(len(mazes)*0.99)
    x = []
    y = []
    for maze in tqdm(mazes):
        empty = np.stack(np.where(maze==E)).transpose()
        index = np.random.choice(empty.shape[0], 1, False)
        goal  = empty[index][0]

        m = maze.copy()
        m[m == D] = E # map doors to path
        m[goal[0], goal[1]] = G
        actions = np.zeros_like(m) - 1 # initialize gold policy to -1

        # BFS, potential field algorithm for gold planning
        queue   = [goal]
        visited = [goal[0]*23+goal[1]]
        idx     = 0
        while idx < len(queue):
            pos = queue[idx]
            for i, action in enumerate(valid_actions):
                possible_prev_pos = pos + action
                hash_code = possible_prev_pos[0]*23 + possible_prev_pos[1] 
                if is_valid_step(m, pos, possible_prev_pos) and hash_code not in visited:
                    queue.append(possible_prev_pos)
                    visited.append(hash_code)
                    actions[possible_prev_pos[0], possible_prev_pos[1]] = inverse_actions[i]
            idx += 1
        x.append(m)
        y.append(actions)

    data = {
        'train': {
            'mazes': mazes[:split],
            'x'    : x[:split],
            'y'    : y[:split],
        },
        'test': {
            'mazes': mazes[split:],
            'x'    : x[split:],
            'y'    : y[split:],
        }
    }
    np.save("data", data, allow_pickle=True)


class Dataset(data.Dataset):
    def __init__(self, data):
        self.x = data['x']
        self.y = data['y']

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]

###############################################################################
#
# Value Iteration Network for navigation
#
###############################################################################

class Nav(nn.Module):
    def __init__(self):
        super(Nav, self).__init__()
        self.maze_emb = nn.Embedding(3, 2)
        self.maze_emb.weight.data = torch.Tensor([
            [0,0],
            [1,0],
            [0,10],
        ])
        self.maze_emb.requires_grad = False

        dim_h = 150
        n_hat_actions = 8
        self.n_actions = 4
        self.K = 10

        # VIN parameters
        self.encode = nn.Conv2d(2, dim_h, 5, padding=2, bias=True)
        self.r  = nn.Conv2d(dim_h, 1, 1, bias=False)
        self.q  = nn.Conv2d(1, n_hat_actions, 5, padding=2, bias=False)
        self.w  = nn.Parameter(torch.zeros(n_hat_actions, 1, 5, 5)).requires_grad_()
        self.fc = nn.Linear(n_hat_actions, self.n_actions, bias=False)

        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        self.opt = torch.optim.Adam(parameters, lr=3e-4)

    def forward(self, maze):
        """
        params:
            maze: [B, H, W]
        return:
            Q_sa: [B, H, W, A]
        """
        m = self.maze_emb(maze)
        m = m.unsqueeze(1).permute(0, 4, 2, 3, 1).squeeze(-1)
        m = self.encode(m)
        r = self.r(m)
        q = self.q(r)

        for i in range(self.K):
            v = q.max(1, True)[0]
            q = F.conv2d(
                    torch.cat((r,v), 1),
                    torch.cat((self.q.weight, self.w), 1),
                    padding=2)
        q = q.unsqueeze(-1).permute(0, 4, 2, 3, 1).squeeze(1)
        q = self.fc(q)
        return q

    def make_tensor(self, x, is_long=1):
        if is_long:
            return torch.LongTensor(x).to(DEVICE)
        else:
            return torch.Tensor(x).to(DEVICE)

    def update(self, m, a, is_train=True):
        m, a = m.long().to(DEVICE), a.long().to(DEVICE)
        q    = self.forward(m)
        loss = self.criterion(q.view(-1, self.n_actions), a.view(-1))

        if is_train:
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

        return loss.detach().cpu().item()

    def policy(self, m):
        m  = self.make_tensor(m)
        q  = self.forward(m)
        pi = q.max(-1)[1]
        return pi.detach().cpu().squeeze().numpy()

    def save(self, path="./nav.pt"):
        torch.save(self.state_dict(), path)

    def load(self, path="./nav.pt"):
        self.load_state_dict(torch.load(path))

###############################################################################
#
# training and testing
#
###############################################################################

def plot_trajectory(maze, trajectory, goal, iter=-1):
    rgb = np.zeros((*maze.shape, 3))
    rgb[maze == 0] = np.array([0,0,0]) # black wall
    rgb[maze == 1] = np.array([1,1,1]) # white path

    step = 0.5 / MAX_N_STEPS / 2
    base = 0.3
    prev_pos = trajectory[0]
    for i, pos in enumerate(trajectory[1:]):
        middle = (prev_pos+pos) // 2
        rgb[middle[0], middle[1]] = np.array([0,0,base])
        base += step
        rgb[pos[0], pos[1]] = np.array([0,0,base])
        base += step
        prev_pos = pos

    start = trajectory[0]
    rgb[start[0], start[1]] = np.array([0,1,0]) # green start
    if (trajectory[-1] == goal).all():
        rgb[goal[0], goal[1]] = np.array([1,1,0]) # yellow goal if reached
    else:
        rgb[goal[0], goal[1]] = np.array([1,0,0]) # red goal if not
    plt.imshow(rgb)
    plt.savefig(IMG_FOLDER+"traj_%d.png" % iter)
    plt.close()

def evaluate(loader, mazes, nav):
    nav.eval()
    with torch.no_grad():
        loss = 0
        for x, y in loader:
            loss += nav.update(x, y, is_train=False)
        loss /= len(loader)

        sr = 0
        for maze in mazes:
            empty = np.stack(np.where(maze==E)).transpose()
            index = np.random.choice(empty.shape[0], 2, False)
            start, goal = empty[index]

            m = maze.copy()
            m[m == D] = E
            m[goal[0], goal[1]] = G
            pi = nav.policy(m.reshape(1, *m.shape))

            done = False
            step = 0
            while step < MAX_N_STEPS:
                step += 1
                action = valid_actions[pi[start[0], start[1]]]
                start += action
                if (start == goal).all():
                    done = True
                    break
                if m[start[0], start[1]] == W:
                    done = False
                    break
            if done:
                sr += 1
        sr /= len(mazes)

    nav.train()
    return loss, sr
    
def plot(stats):
    xtr = stats['tr_i']
    ytr = stats['tr_l']
    xte = stats['te_i']
    yte = stats['te_l']
    ysr = stats['te_sr']
    best_loss = stats['best_loss']
    highest_sr = stats['highest_sr']
    plt.figure()
    ax1 = plt.subplot(211)
    plt.title("vin navigation training curve")
    plt.plot(xtr, ytr, 'r', label="train loss")
    plt.plot(xte, yte, 'm', label="test loss")
    ax1.axhline(y=best_loss, color='k', label=("%.2f" % best_loss))
    ax1.legend()
    plt.grid(True)
    ax2 = plt.subplot(212)
    plt.plot(xte, ysr, 'c', label="test success rate")
    ax2.axhline(y=highest_sr, color='k', label=("%.2f" % highest_sr))
    ax2.legend()
    plt.grid(True)
    plt.savefig(IMG_FOLDER+"nav_train_curve.png")
    plt.close()

def is_valid_step(maze, from_pos, to_pos):
    if max(to_pos) > 22:
        return False
    if min(to_pos) < 0:
        return False
    if maze[to_pos[0], to_pos[1]] == W:
        return False
    door = (from_pos+to_pos)//2
    if maze[door[0], door[1]] == W:
        return False
    return True
    
def train(n_epochs=10000, pretrained_path=""):
    d = np.load("data.npy", allow_pickle=True)[()]
    trainset, testset = Dataset(d['train']), Dataset(d['test'])
    params = {'batch_size': 64, 'shuffle':True, 'num_workers':6}
    train_loader = data.DataLoader(trainset, **params)
    test_loader  = data.DataLoader(testset, **params)

    nav = Nav().to(DEVICE)
    if pretrained_path is not "":
        try:
            nav.load(pretrained_path)
        except:
            print("[ERROR] failed to load model from %s" % pretrained_path)

    stats = {
        'best_loss' : np.inf,
        'highest_sr': 0,      # success rate
        'tr_i' : [],
        'tr_l' : [],
        'te_i' : [],
        'te_l' : [],
        'te_sr': [],
    }
    for epoch in range(1, n_epochs+1):
        loss = 0
        for x, y in train_loader:
            loss += nav.update(x, y)
        loss /= len(train_loader)
        stats['tr_i'].append(epoch)
        stats['tr_l'].append(loss)

        print("[INFO] loss %s" % loss)
        if epoch % 5 == 0:
            l, sr = evaluate(test_loader, d['test']['mazes'], nav)
            stats['te_i'].append(epoch)
            stats['te_l'].append(l)
            stats['te_sr'].append(sr)

            if sr > stats['highest_sr']:
                stats['highest_sr'] = sr
            if l < stats['best_loss']:
                stats['best_loss'] = l
                nav.save()
            plot(stats)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == 'g':
            generate_data()
        elif sys.argv[1] == 'p':
            train(pretrained_path=sys.argv[2])
        elif sys.argv[1] == 't':
            test(pretrained_path=sys.argv[2])
        else:
            print("[ERROR] unknown flag ('g' for generate data, 'v' for visualize data, 'p' for continue training with pretrained model)")
    else:
        train()