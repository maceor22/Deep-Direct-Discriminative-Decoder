from copy import deepcopy
import scipy.io as sio
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
import numpy as np
import mat73

# plot maze data for a given rat and session
def plot_maze(name, session, Fs, n_points):
    # name: name of rat
    # session: session to be plotted
    # Fs: sampling frequency
    # n_points: number of points to be included
    
    # load spike and position data specific to rat name and session
    if name == 'Remy':        
        spikesFile = sio.loadmat(f'Datasets/SunMaze/Remy/remymarks{session[0]}.mat')['marks'][0][-1][0][session[1]][0]
        rem = [0,4,7,15,17,26]
        
        posRaw = sio.loadmat(f'Datasets/SunMaze/Remy/remypos{session[0]}.mat')['pos'][0][-1][0][session[1]][0][0][3][:,1:5]
        
    elif name == 'Jaq':
        spikesFile = sio.loadmat(f'Datasets/SunMaze/Jaq/Jaqspikes0{session[0]}.mat')['spikes'][0][-1][0][session[1]][0]
        rem = [12,18]
        
        if session[0] <= 2:
            posRaw = sio.loadmat(f'Datasets/SunMaze/Jaq/Jaqpos0{session[0]}.mat')['pos'][0][-1][0][session[1]][0][0][0][:,1:5]
        else:
            posRaw = sio.loadmat(f'Datasets/SunMaze/Jaq/Jaqposdlc0{session[0]}.mat')['posdlc'][0][-1][0][session[1]][0][0][0][:,1:3]
            posRaw = np.delete(posRaw, np.where(np.isnan(posRaw)), axis = 0)
        
    elif name == 'Bon':
        spikesFile = mat73.loadmat(f'Datasets/SunMaze/Bon/Bonspikes0{session[0]}.mat')
        print(spikesFile)
# =============================================================================
#         with h5py.File(f'Datasets/SunMaze/Bon/bonpos0{session[0]}.mat', 'r') as f:
#             posRaw = open(f['pos'])
# =============================================================================
    
    
    posRaw -= posRaw.min(axis = 0)
    if n_points != 'all':
        posRaw = posRaw[:n_points,:]    
    
    # plotting
    plt.figure()
    plt.plot(posRaw[:,0], posRaw[:,1], '0.4')
    
    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    plt.title('Maze Shape (X-Y) view')
    
    
    dt = 0.033
    Time = np.arange(posRaw.shape[0]) * dt
    
    fig, ax = plt.subplots(2, 1, sharex = True)
    
    plt.xlabel('Time [s]')
    
    ax[0].plot(Time, posRaw[:,0], 'k')
    ax[0].set_ylabel('X position')
    
    ax[1].plot(Time, posRaw[:,1], 'k')
    ax[1].set_ylabel('Y position')
    
    fig.suptitle('Trajectory versus Time')

    
    
# class used to loading and holding maze data (spike data and position data)
class Maze(object):
    
    def __init__(self, 
                 name, session, n_points, include_velocity = False, dirvel = True,
                 rem_insig_chans = False, threshold = None,
                 ):
        # name: name of rat
        # session: session to be plotted
        # Fs: sampling frequency
        # rem_insig_chans: boolean indicating whether to remove insignificant channels
        # threshold: spike frequency threshold to determine significance of spike channel
        
        self.info = {
            'name' : name,
            'session' : session,
            'n_points' : n_points,
            }
        
        # load spike and position data for given rat name and session
        if name == 'Remy':        
            spikesFile = sio.loadmat(f'Datasets/SunMaze/Remy/remymarks{session[0]}.mat')['marks'][0][-1][0][session[1]][0]
            rem = [0,4,7,15,17,26]
            
            posRaw = sio.loadmat(f'Datasets/SunMaze/Remy/remypos{session[0]}.mat')['pos'][0][-1][0][session[1]][0][0][3][:,:5]
            
            ntrodes = [i for i in range(30) if i not in rem]
            dat = []
            for n in ntrodes:
                dat.append(spikesFile[n][0][0][-2].reshape(-1))
        
        elif name == 'Jaq':
            spikesFile = sio.loadmat(f'Datasets/SunMaze/Jaq/Jaqspikes0{session[0]}.mat')['spikes'][0][-1][0][session[1]][0]
            rem = [12,18]
# =============================================================================
#             for i in spikesFile:
#                 print(i[0][0][0], i.shape)
# =============================================================================
# =============================================================================
#             inq = spikesFile[1]
#             print(inq, inq.shape)
# =============================================================================
            idx = 0            
            inq = []
            for s in spikesFile:
                s = s.reshape(-1)
                #print(s.shape)
                for i in s:
                    if i.shape == (1,1):
                        print(idx, len(i[0][0]), i[0][0])
                        #inq.append(i[0][0][0])
                        #print('    ', i[0][0][0].shape)
                        idx += 1
            
            for i in inq:
                print('%.2f, %.2f,' % (i.min(), i.max()), i[0].shape)
                #print(i)
            print(len(inq))
# =============================================================================
#             for i in inq:
#                 print(i[1])
# =============================================================================
            

# =============================================================================
#             inq = spikesFile[24][0]#[4][0][0][4]#[4]#[0][0][0][0]
#             print(inq, inq.shape)
#             
#             for i in inq:
#                 print(i.shape)
#                 if i.shape == (1,1):
#                     print(i[0][0][4][0][0])
# =============================================================================
            
            
            if session[0] <= 2:
                posRaw = sio.loadmat(f'Datasets/SunMaze/Jaq/Jaqpos0{session[0]}.mat')['pos'][0][-1][0][session[1]][0][0][0][:,:5]
            else:
                posRaw = sio.loadmat(f'Datasets/SunMaze/Jaq/Jaqposdlc0{session[0]}.mat')['posdlc'][0][-1][0][session[1]][0][0][0][0,:]
                print(posRaw, posRaw.shape)
        
        elif name == 'Bon':
            spikesFile = sio.loadmat(f'Datasets/SunMaze/Bon/bonspikes0{session[0]}.mat')['spikes'][0][-1][0][session[1]][0]#[11]#[0][0]#[0][0][0][:,-1]
            #print(spikesFile, spikesFile.shape)
            
            rem = [10, 20, 26, 27, 31]
            rem += list(range(38, 100))
            #print(rem)
            ix = 0
            dat = []
            for f in spikesFile:
                f = f[0]
                if f.shape[0] != 0:
                    for c in f:
                        if c.shape == (1,1):
                            c = c[0][0]
                            #print(ix, c[0].shape)
                            if ix not in rem:
                                dat.append(c[0][:,0])
                            ix += 1
                            
            posRaw = mat73.loadmat(f'Datasets/SunMaze/Bon/bonpos0{session[0]}.mat')['pos'][-1][session[1]]['data'][:,:5]
        
        
        elif name == 'Emile':
            spikesFile = mat73.loadmat(f'Datasets/SunMaze/Emile/emilespikes{session[0]}.mat')['spikes'][-1][session[1]]#[2][0]
            
            #rem = [14, 25, 79, 91]
            rem = [14, 25, 79, 91]
            # 6, 18, 26, 50, 52, 53, 56, 58, 61, 63
            """ rem = [11, 14, 15, 25, 26, 28, 30, 31, 36, 
                   37, 39, 40, 43, 45, 48, 50, 54, 
                   55, 56, 58, 60, 61, 62, 63, 68, 
                   74, 77, 78, 79, 84, 88, 91, 93,]
            rem = [6, 11, 14, 15, 25, 26, 28, 30, 31, 36, 
                   37, 39, 40, 43, 45, 48, 50, 54, 
                   55, 56, 58, 60, 61, 62, 63, 68, 
                   74, 77, 78, 79, 84, 88, 91, 93,] """
            dat = []
            for i in range(len(spikesFile)):
                s = spikesFile[i]
                if s != None:
                    for j in range(len(s)):
                        #print(i, j, type(s[j]), len(s[j]))
                        #if type(s[j]['data']) == type(None):
                        dat.append(s[j]['data'])
            dat = [dat[i] for i in range(len(dat)) if i not in rem]
            
# =============================================================================
#             for i in range(len(dat)):
#                 print(i, type(dat[i]), dat[i].shape if type(dat[i]) == type(np.array(0)) else '')
# =============================================================================
                
            posRaw = sio.loadmat(f'Datasets/SunMaze/Emile/emilepos{session[0]}.mat')['pos'][0][-1][0][session[1]][0][0][3][:,:5]
            #print(posRaw, posRaw.shape)
# =============================================================================
#             for p in posRaw:
#                 print(p, p.shape)
# =============================================================================
            
        
        if n_points != 'all':
            posRaw = posRaw[:n_points,:]
        
        #posRaw[:,1:] -= posRaw[:,1:].min(axis = 0)
        
        # basic data cleaning to yield spike train data from spike times
        spikes = torch.zeros((posRaw.shape[0],len(dat)))
        
        for j in range(len(dat)):
            idx = 0
            for i in range(posRaw.shape[0]):
                count = 0
                done = False
                while not done:
                    #print(dat[j].shape)
                    if idx >= dat[j].shape[0]:
                        done = True
                    elif dat[j][idx] < posRaw[i,0]:
                        idx += 1
                        count += 1
                    else:
                        done = True
                spikes[i,j] = count


        if rem_insig_chans:
            if threshold == None:
                threshold = 100
            
            spikes = torch.cat([
                spikes[:,i].unsqueeze(1) for i in range(spikes.size(1))
                if spikes[:,i].sum() >= threshold
                ], dim = 1)
        
        self.spike_times = dat
        self.spike_counts = spikes
        self.pos = torch.from_numpy(posRaw[:,1:])
        
        if include_velocity:
            if not dirvel:
                xvel = torch.cos(self.pos[:,2]) * self.pos[:,3]
                yvel = torch.sin(self.pos[:,2]) * self.pos[:,3]
                self.pos[:,2], self.pos[:,3] = xvel, yvel
        else:
            self.pos = self.pos[:,:2]
        
    
    # method for downsampling position data
    def downsample_position(self, bin_size = 10, overlap = 0):
        # bin_size: size of bin to use when downsampling
        # overlap: overlap between sequential bins during downsampling
        
        lower = 0
        upper = bin_size
        step = bin_size - overlap
        
        data_avg = []
        data_var = []
        
        # loop for downsampling
        while upper <= self.posRaw.size(dim = 0):
            # save mean and variance of data in downsampled bin
            data_avg.append(self.posRaw[lower:upper,:].mean(dim = 0).unsqueeze(dim = 0))
            data_var.append(self.posRaw[lower:upper,:].var(dim = 0, unbiased = True).unsqueeze(dim = 0))
        
            lower += step
            upper += step
        
        # concatenate bin means and bin vars
        new_data = torch.cat([
            torch.cat(data_avg, dim = 0),
            torch.cat(data_var, dim = 0),
            ], dim = 1).unsqueeze(-1)
        
        return new_data
        
    
    # method for generating position history given raw data
    def generate_position_history(self, history_length = 0):
        pos = self.pos.unsqueeze(-1)
        
        history_data = []
        # loop for aggregating position history
        for i in range(history_length, pos.size(dim = 0)):
            # get current timestep and history, then reshape and save to list
            new = pos[i-history_length:i,:,:].transpose(0,-1).transpose(1,-1)
            history_data.append(new)
        
        # concatenate position history data and return
        return torch.cat(history_data, dim = 0).float()
        
    
    def generate_data(
            self, pos_history_length = 0, 
            spike_history_length = 0, spike_bin_size = 1, 
            label_history_length = 0, shuffle = False,
        ):
        lower_index = max(
            pos_history_length, spike_history_length+1, label_history_length,
            )
            
        spike_hist = torch.cat([
            self.spike_counts[i-spike_history_length:i+1,:].unsqueeze(0) 
            for i in range(lower_index, self.spike_counts.size(0))
            ], dim = 0)
        
        pos_data = torch.cat([
            self.pos[i-(pos_history_length+1):i,:].unsqueeze(0) 
            for i in range(lower_index, self.pos.size(0))
            ], dim = 0)
        
        if spike_bin_size == 1:
            spike_data = spike_hist
        else:
            num_bins = int(spike_history_length/spike_bin_size)
            spike_data = []
            for i in range(spike_hist.size(0)):
                dat = spike_hist[i,:,:]
                new = []
                lower = 0
                upper = spike_bin_size
                for _ in range(num_bins):
                    new.append(dat[lower:upper,:].sum(0).unsqueeze(0))
                new.append(dat[-1,:].unsqueeze(0))
                new = torch.cat(new, dim = 0).unsqueeze(0)
                spike_data.append(new)
            spike_data = torch.cat(spike_data, dim = 0)
                        
        if label_history_length == 0:
            label_data = self.pos[lower_index:,:]
        else:
            label_data = torch.cat([
                self.pos[i-label_history_length:i+1,:].unsqueeze(0) 
                for i in range(lower_index, self.pos.size(0))
                ], dim = 0)
        
        if shuffle:
            order = torch.randperm(spike_data.size(0))
            spike_data = spike_data[order]
            pos_data = pos_data[order]
            label_data = label_data[order]
                
        return spike_data, pos_data, label_data
    
    
    # method for plotting heat map of spike train data
    def plot_spikes_heat_map(self, directory = None):
        spikes = self.spikes.numpy()
                
        Fs = self.info['Fs']
        ntrode = np.arange(0, spikes.shape[0], 1)
        tstep  = np.arange(0, spikes.shape[1], 1) / Fs
        
        fig, ax = plt.subplots()
        ax.pcolormesh(tstep, ntrode, spikes)
        plt.xlabel('Time [s]')
        plt.ylabel('Ntrode #')
        plt.title(f'Spikes over time | Time resolution: {Fs} Hz')
        plt.show()
        
    
    # method for plotting position data
    def plot_position(self, plot_map = False, plot_over_time = False, directory = None):
        x_pos, y_pos = self.pos[:,0].numpy(), self.pos[:,1].numpy()
        n_points = x_pos.shape[0]
        
        # plot x-y view map
        if plot_map:
            plt.figure()
            idx = [0, n_points*.1, n_points*.2, n_points*.3, n_points*.4, n_points*.5, 
                   n_points*.6, n_points*.7, n_points*.8, n_points*.9, n_points-1]
            
            Fs = 30
            
            plt.plot(x_pos, y_pos, '0.4')
            for i in idx:
                plt.text(
                    x_pos[int(i)], y_pos[int(i)], 
                    't=%.3fs' % (int(i)/Fs),
                    fontsize = 8, color = 'blue'
                    )
            name, session = self.info['name'], self.info['session']
            plt.xlabel('X axis')
            plt.ylabel('Y axis')
            plt.title(f'{name} Position (X-Y view) | Session: {session}')
            plt.show()
        
        # plot x and y independently versus time
        if plot_over_time:
            fig, ax = plt.subplots(2, 1, sharex = True)
            Time = np.arange(0,n_points)/Fs
            
            ax[0].plot(Time, x_pos, 'blue')
            ax[1].plot(Time, y_pos, 'orange')
        
            ax[0].set_ylabel('X Position')
            ax[1].set_ylabel('Y Position')
            
            plt.xlabel('Time [s]')
            fig.suptitle('Rat Position vs Time')
            plt.show()
            

# custom Dataset class used during model training
class Data(Dataset):
    
    def __init__(self, inputs, labels):
        self.x = inputs
        self.y = labels
        
    def __len__(self):
        return self.x.size(dim = 0)

    def __getitem__(self, index):
        xi = self.x[index,:]
        yi = self.y[index]
        return xi, yi
    

# class for applying range normalize transform to position data
class RangeNormalize(object):
    
    def __init__(self, dim, norm_mode = 'auto', velocity_data = False):
        if norm_mode == 'auto':
            self.norm_mode = [0 for i in range(dim)]
        else:
            self.norm_mode = norm_mode
        self.velocity_data = velocity_data
        
    # method for fitting transform object
    def fit(self, data = None, range_min = None, range_max = None):
        if data is not None:
            if data.dim() == 2:
                data = deepcopy(data).unsqueeze(1)
                
# =============================================================================
#             if self.velocity_data:        
#                 self.vel_mean = data[:,-1,2:].mean(dim = 0)
#                 self.vel_std  = data[:,-1,2:].std(dim = 0)
# =============================================================================
            
            if range_min is None:
                self.range_min = data[:,-1,:].min(dim = 0)[0]
            else:
                self.range_min = torch.tensor(range_min)
            if range_max is None:
                self.range_max = data[:,-1,:].max(dim = 0)[0]
            else:
                self.range_max = torch.tensor(range_max)
        else:
            self.range_min = torch.tensor(range_min)
            self.range_max = torch.tensor(range_max)
        
        self.loc = torch.zeros_like(self.range_min)
        scale = torch.ones_like(self.range_min)
        for i in range(self.loc.size(0)):
            if self.norm_mode[i] == 0:
                self.loc[i] = self.range_min[i]
                
            elif self.norm_mode[i] == 1:
                self.loc[i] = (self.range_min[i]+self.range_max[i])/2
                scale[i] = 2
        
        self.scale = (self.range_max - self.range_min) / scale
        
            
    # method for applying transform to position data
    def transform(self, data):
        new_data = deepcopy(data.detach())
        
        if data.dim() == 2:
            new_data = new_data.unsqueeze(1)
        
# =============================================================================
#         if self.velocity_data:
#             new_data[:,:,:2] = (new_data[:,:,:2] - self.range_min) / (self.range_max - self.range_min)
#             new_data[:,:,2:] = (new_data[:,:,2:] - self.vel_mean) / self.vel_std
#         else:
# =============================================================================
        new_data = (new_data - self.loc) / self.scale
        
        if data.dim() == 2:
            new_data = new_data.squeeze(1)
        return new_data
    
    # method that runs fit and transform on position data
    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)
    
    # method for untransforming transformed data
    def untransform(self, data, variance = None):
        new_data = deepcopy(data.detach())
        
        if data.dim() == 2:
            new_data = new_data.unsqueeze(1)
        
# =============================================================================
#         if self.velocity_data:
#             new_data[:,:,:2] = new_data[:,:,:2] * (self.range_max - self.range_min) + self.range_min
#             new_data[:,:,2:] = new_data[:,:,2:] * self.vel_std + self.vel_mean
#         else:
# =============================================================================
        new_data = new_data * self.scale + self.loc
        
        if data.dim() == 2:
            new_data = new_data.squeeze()
        
        if variance == None:
            return new_data
        
        else:            
            variance *= self.scale
# =============================================================================
#             if self.velocity_data:
#                 variance[:,:,2:] *= self.vel_std
# =============================================================================
            return new_data, variance.squeeze()
        
        
# class for applying one type of arm-dist transform to position data
class WMazeDiscreteTransform1(object):
    
    # optionally range normalize position data before fitting and transforming
    def __init__(self, range_norm = False):
        self.range_norm = range_norm
        
    # method for fitting transform object
    def fit(self, data):        
        self.center = torch.zeros((4,))
        c = [[],[],[],[]]
        
        for i in range(data.size(0)):
            if data[i,-1,1] > 80:
                c[0].append(data[i,-1,1].item())
            else:
                if data[i,-1,0] < 30:
                    c[1].append(data[i,-1,0].item())
                elif data[i,-1,0] < 70:
                    c[2].append(data[i,-1,0].item())
                else:
                    c[3].append(data[i,-1,0].item())
        
        if self.range_norm:
            self.range = torch.zeros((4,2))
            for i in range(4):
                #c = torch.Tensor(c[i])
                self.center[i] = torch.tensor(c[i]).mean()
                mi = min(c[i])
                ma = max(c[i])
                self.range[i,0] = ma - mi
            self.range[0,1] = 100
            self.range[1:,1] = 80
        
        else:
            for i in range(4):
                self.center[i] = torch.tensor(c[i]).mean()
            
    # method for applying transform to position data
    def transform(self, data):
        new_data = torch.zeros((data.size(0), data.size(1),6))
        
        for i in range(data.size(0)):
            for j in range(data.size(1)):
                
                if data[i,j,1] > 80:
                    new_data[i,j,0] = 1
                    new_data[i,j,4] = self.center[0] - data[i,j,1]
                    new_data[i,j,5] = data[i,j,0]
                    
                    if self.range_norm:
                        new_data[i,j,4] /= self.range[0,0]
                        new_data[i,j,5] /= self.range[0,1]
                    
                else:
                    if data[i,j,0] < 30:
                        new_data[i,j,1] = 1
                        new_data[i,j,4] = self.center[1] - data[i,j,0]
                        new_data[i,j,5] = 80 - data[i,j,1]
                        
                        if self.range_norm:
                            new_data[i,j,4] /= self.range[1,0]
                            new_data[i,j,5] /= self.range[1,1]
                    
                    elif data[i,j,0] < 70:
                        new_data[i,j,2] = 1
                        new_data[i,j,4] = self.center[2] - data[i,j,0]
                        new_data[i,j,5] = 80 - data[i,j,1]
                        
                        if self.range_norm:
                            new_data[i,j,4] /= self.range[2,0]
                            new_data[i,j,5] /= self.range[2,1]
                    
                    else:
                        new_data[i,j,3] = 1
                        new_data[i,j,4] = self.center[3] - data[i,j,0]
                        new_data[i,j,5] = 80 - data[i,j,1]
                        
                        if self.range_norm:
                            new_data[i,j,4] /= self.range[3,0]
                            new_data[i,j,5] /= self.range[3,1]
        
        return new_data
                
    # method that runs fit and transform on position data
    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)
    
    # method for untransforming transformed position data
    def untransform(self, data, variance = None):
        new_data = torch.zeros((data.size(0), data.size(1), 2))
        
        for i in range(data.size(0)):
            for j in range(data.size(1)):
                
                if data[i,j,0] == 1:
                    if self.range_norm:
                        new_data[i,j,0] = data[i,j,5] * self.range[0,1]
                        new_data[i,j,1] = self.center[0] - (self.range[0,0] * data[i,j,4])
                    else:
                        new_data[i,j,0] = data[i,j,5]
                        new_data[i,j,1] = self.center[0] - data[i,j,4]
                
                elif data[i,j,1] == 1:
                    if self.range_norm:
                        new_data[i,j,0] = self.center[1] - (self.range[1,0] * data[i,j,4])
                        new_data[i,j,1] = 80 - (data[i,j,5] * self.range[1,1])
                    else:
                        new_data[i,j,0] = self.center[1] - data[i,j,4]
                        new_data[i,j,1] = 80 - data[i,j,5]
                    
                elif data[i,j,2] == 1:
                    if self.range_norm:
                        new_data[i,j,0] = self.center[2] - (self.range[2,0] * data[i,j,4])
                        new_data[i,j,1] = 80 - (data[i,j,5] * self.range[2,1])
                    else:
                        new_data[i,j,0] = self.center[2] - data[i,j,4]
                        new_data[i,j,1] = 80 - data[i,j,5]
        
                else:
                    if self.range_norm:
                        new_data[i,j,0] = self.center[3] - (self.range[3,0] * data[i,j,4])
                        new_data[i,j,1] = 80 - (data[i,j,5] * self.range[3,1])
                    else:
                        new_data[i,j,0] = self.center[3] - data[i,j,4]
                        new_data[i,j,1] = 80 - data[i,j,5]
                    
                    
        if variance == None:
            return new_data
        else:
            new_var = torch.zeros((variance.size(0), 2))
            for i in range(variance.size(0)):
                
                if data[i,-1,0] == 1:
                    if self.range_norm:
                        new_var[i,0] = variance[i,1] * self.range[0,1]
                        new_var[i,1] = variance[i,0] * self.range[0,0]
                    else:
                        new_var[i,0] = variance[i,1]
                        new_var[i,1] = variance[i,0]
                
                elif data[i,-1,1] == 1 and self.range_norm:
                    new_var[i,0] = variance[i,0] * self.range[1,0]
                    new_var[i,1] = variance[i,1] * self.range[1,1]
            
                elif data[i,-1,2] == 1 and self.range_norm:
                    new_var[i,0] = variance[i,0] * self.range[2,0]
                    new_var[i,1] = variance[i,1] * self.range[2,1]
                    
                elif data[i,-1,3] == 1 and self.range_norm:
                    new_var[i,0] = variance[i,0] * self.range[3,0]
                    new_var[i,1] = variance[i,1] * self.range[3,1]
                    
                else:
                    new_var[i,0] = variance[i,0]
                    new_var[i,1] = variance[i,1]
            
            return new_data, new_var
    
    
# class for applying a second type of arm-dist transform to position data
class WMazeDiscreteTransform2(object):
    
    # optionally range normalize position data before fitting and applying transform
    def __init__(self, range_norm = False):
        self.range_norm = range_norm
        
    # method for fitting transform object
    def fit(self, data):
        if data.dim() == 2:
            data = deepcopy(data).unsqueeze(1)
        
        self.center = torch.zeros((4,))
        c = [[],[],[],[]]
        
        for i in range(data.size(0)):
            if data[i,-1,1] > 80:
                c[0].append(data[i,-1,1].item())
            else:
                if data[i,-1,0] < 30:
                    c[1].append(data[i,-1,0].item())
                elif data[i,-1,0] < 70:
                    c[2].append(data[i,-1,0].item())
                else:
                    c[3].append(data[i,-1,0].item())
        
        if self.range_norm:
            self.range = 100
        
        for i in range(4):
            self.center[i] = torch.tensor(c[i]).mean()
            
    # method for transforming position data
    def transform(self, data):
        if data.dim() == 2:
            data = deepcopy(data).unsqueeze(1)
        
        m = torch.zeros((4,))
        m[0] = -1
        m[1] = 1
        m[2] = -1
        m[3] = 1
        
        b = torch.zeros((4,))
        b[0] = self.center[0] - m[0] * self.center[1]
        b[1] = self.center[0] - m[1] * self.center[2]
        b[2] = self.center[0] - m[2] * self.center[2]
        b[3] = self.center[0] - m[3] * self.center[3]
        
        new_data = torch.zeros((data.size(0), data.size(1),6))
        
        for i in range(data.size(0)):
            for j in range(data.size(1)):
                
                boolean0 = data[i,j,1]<=m[0]*data[i,j,0]+b[0]
                boolean1 = (data[i,j,1]<=m[1]*data[i,j,0]+b[1]) and (data[i,j,1]<=m[2]*data[i,j,0]+b[2])
                boolean2 = data[i,j,1]<=m[3]*data[i,j,0]+b[3]
                
                if data[i,j,0] < 30 and boolean0:
                    new_data[i,j,2] = 1
                    new_data[i,j,5] = (self.center[0] - data[i,j,1]) + (self.center[2] - self.center[1])
                
                elif data[i,j,0] < 70 and boolean1:
                    new_data[i,j,3] = 1
                    new_data[i,j,5] = self.center[0] - data[i,j,1]
                
                elif data[i,j,0] >= 70 and boolean2:
                    new_data[i,j,4] = 1
                    new_data[i,j,5] = (self.center[0] - data[i,j,1]) + (self.center[3] - self.center[2])
                        
                elif data[i,j,0] < self.center[2]:
                    new_data[i,j,0] = 1
                    new_data[i,j,5] = self.center[2] - data[i,j,0]
                
                else:
                    new_data[i,j,1] = 1
                    new_data[i,j,5] = data[i,j,0] - self.center[2]
        
        if self.range_norm:
            new_data[:,:,5] /= self.range
        
        return new_data.squeeze()
                
    # method that runs fit and transform on position data
    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)
    
    # method for untransforming transformed position data
    def untransform(self, data, variance = None):
        if data.dim() == 2:
            data = deepcopy(data.detach()).unsqueeze(1)
        
        new_data = torch.zeros((data.size(0), data.size(1), 2))
        
        for i in range(data.size(0)):
            for j in range(data.size(1)):
                
                if data[i,j,0] == 1:
                    if self.range_norm:
                        new_data[i,j,0] = self.center[2] - (self.range * data[i,j,5])
                    else:
                        new_data[i,j,0] = self.center[2] - data[i,j,5]                    
                    new_data[i,j,1] = self.center[0]
                
                elif data[i,j,1] == 1:
                    if self.range_norm:
                        new_data[i,j,0] = self.center[2] + (self.range * data[i,j,5])
                    else:
                        new_data[i,j,0] = self.center[2] + data[i,j,5]
                    new_data[i,j,1] = self.center[0]
                    
                elif data[i,j,2] == 1:
                    new_data[i,j,0] = self.center[1]
                    if self.range_norm:
                        new_data[i,j,1] = self.center[0] - (self.range * data[i,j,5] - (self.center[2] - self.center[1]))
                    else:
                        new_data[i,j,1] = self.center[0] - (data[i,j,5] - (self.center[2] - self.center[1]))
        
                elif data[i,j,3] == 1:
                    new_data[i,j,0] = self.center[2]
                    if self.range_norm:
                        new_data[i,j,1] = self.center[0] - (self.range * data[i,j,5])
                    else:
                        new_data[i,j,1] = self.center[0] - data[i,j,5]
                        
                elif data[i,j,4] == 1:
                    new_data[i,j,0] = self.center[3]
                    if self.range_norm:
                        new_data[i,j,1] = self.center[0] - (self.range * data[i,j,5] - (self.center[3] - self.center[2]))
                    else:
                        new_data[i,j,1] = self.center[0] - (data[i,j,5] - (self.center[3] - self.center[2]))
                    
                    
        if variance == None:
            return new_data.squeeze()
        else:
            new_var = torch.zeros((variance.size(0), 2))
            for i in range(variance.size(0)):
                
                if data[i,-1,0] == 1 or data[i,-1,1] == 1:
                    new_var[i,0] = variance[i]
                    
                else:
                    new_var[i,1] = variance[i]
                
            if self.range_norm:
                new_var *= self.range
            
            return new_data.squeeze(), new_var
    

# Bon:
#   line1_intercept: -100 | line2_intercept: 330
#   bound1: 195 | bound2: 235

# class for applying a third transform specialized for the W-maze datasets
class WMazeTransform(object):
    
    # optionally range normalize position data before fitting and applying transform
    def __init__(
            self, 
            line1_intercept, line2_intercept, 
            x1_bound, x2_bound,
            range_normalize = False,
            xmin = None, xmax = None, ymin = None, ymax = None, 
            ):
        self.x = (xmin, xmax)
        self.y = (ymin, ymax)
        self.m = (1, -1)
        self.b = (line1_intercept, line2_intercept)
        self.bound = (x1_bound, x2_bound)
        self.norm = range_normalize
        
        if range_normalize:
            self.range = torch.tensor([xmax-xmin,ymax-ymin])
            self.range_min = torch.tensor([xmin,ymin])
        
    def rotate(self, point, m, dy):
        return torch.cat([(point[0]-m*dy).view(-1), (point[1]-dy).view(-1)], dim = 0)
    
    # method for transforming position data
    def transform(self, data):
        data = deepcopy(data)
        if data.dim() == 2:
            data = data.unsqueeze(1)
        
        for i in range(data.size(0)):
            for j in range(data.size(1)):
                dy = -1
                if data[i,j,0] < self.bound[0]:
                    m = self.m[0]
                    dy = data[i,j,1] - (self.m[0] * data[i,j,0] + self.b[0])
                elif data[i,j,0] > self.bound[1]:
                    m = self.m[1]
                    dy = data[i,j,1] - (self.m[1] * data[i,j,0] + self.b[1])
                
                if dy > 0:
                    data[i,j,:] = self.rotate(data[i,j,:], m, dy)
        
        if self.norm:
            data[:,:,0] = (data[:,:,0] - self.range_min[0]) / self.range[0]
            data[:,:,1] = (data[:,:,1] - self.range_min[1]) / self.range[1]
        
        return data.squeeze()
    
    def unrotate(self, point, m, dx):
        return torch.cat([(point[0]+m*dx).view(-1), (point[1]+dx).view(-1)], dim = 0)
        
    # method for untransforming transformed position data
    def untransform(self, data, variance = None):
        if data.dim() == 2:
            data = deepcopy(data.detach()).unsqueeze(1)
        if variance is not None:
            variance = deepcopy(variance.detach())
            if variance.dim() == 2:
                variance = variance.unsqueeze(1)
        
        if self.norm:
            data[:,:,0] = (data[:,:,0] * self.range[0]) + self.range_min[0]
            data[:,:,1] = (data[:,:,1] * self.range[1]) + self.range_min[1]
        
        for i in range(data.size(0)):
            for j in range(data.size(1)):
                dx = -1
                if data[i,j,0] < self.bound[0]:
                    m = self.m[0]
                    dx = ((data[i,j,1] - self.b[0]) / self.m[0]) - data[i,j,0]
                elif data[i,j,0] > self.bound[1]:
                    m = self.m[1]
                    dx = data[i,j,0] - ((data[i,j,1] - self.b[1]) / self.m[1])
                
                if dx > 0:
                    data[i,j,:] = self.unrotate(data[i,j,:], m, dx)
                    if variance is not None:
                        variance[i,j,:] = variance[i,j,:].flip(0)
        
        if self.norm and variance is not None:
            variance[:,:,0] *= self.range[0]
            variance[:,:,1] *= self.range[1]
                    
        if variance == None:
            return data.squeeze()
        else:
            
            return data.squeeze(), variance.squeeze()
    

# =============================================================================
# wm = Maze(
#     name = 'Jaq', 
#     session = (1,3),
#     n_points = 'all', 
#     )
# 
# print(len(wm.spike_times), wm.spike_counts.size(), wm.pos.size())
# =============================================================================


# =============================================================================
# wm = Maze(
#     name = 'Remy', 
#     session = (35,1), 
#     n_points = 'all', 
#     )
# =============================================================================




    
