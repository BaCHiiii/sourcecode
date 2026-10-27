import numpy as np
import torch
class SpectrumTransfer:
    
    def __init__(self):
        self.extend_mode = 0
        self.B = None
        self.C = None
        self.M = None
        self.TransM = None
        self.TransMin = 0.0
        self.TransMax = 1.0
        
    def get_extend(self, src_rgb):
        
        extend = None

        if self.extend_mode == 0:
            extend = np.array([src_rgb[:,0], src_rgb[:,1], src_rgb[:,2], torch.pow(src_rgb[:,0],2), torch.pow(src_rgb[:,1],2), torch.pow(src_rgb[:,2],2)])
        elif self.extend_mode == 1:
            extend = np.array([src_rgb[:,0], src_rgb[:,1], src_rgb[:,2], torch.pow(src_rgb[:,0],2), torch.pow(src_rgb[:,1],2), torch.pow(src_rgb[:,2],2),src_rgb[:,0]* src_rgb[:,1],src_rgb[:,0]* src_rgb[:,2],src_rgb[:,1]* src_rgb[:,2]])
        elif self.extend_mode == 2:
            extend = np.array([src_rgb[:,0], src_rgb[:,1], src_rgb[:,2], torch.pow(src_rgb[:,0],2), torch.pow(src_rgb[:,1],2), torch.pow(src_rgb[:,2],2),src_rgb[:,0]* src_rgb[:,1],src_rgb[:,0]* src_rgb[:,2],src_rgb[:,1]* src_rgb[:,2], torch.pow(src_rgb[:,0],3), torch.pow(src_rgb[:,1],3), torch.pow(src_rgb[:,2],3)])

        return extend
    def get_extend_batch(self, src_rgb):
        
        extend = None

        if self.extend_mode == 0:
            src_rgb = torch.stack(([src_rgb[:,:,0], src_rgb[:,:,1], src_rgb[:,:,2], torch.pow(src_rgb[:,:,0],2), torch.pow(src_rgb[:,:,1],2), torch.pow(src_rgb[:,:,2],2)]), 0)
        elif self.extend_mode == 1:
            src_rgb = torch.stack(([src_rgb[:,0], src_rgb[:,1], src_rgb[:,2], torch.pow(src_rgb[:,0],2), torch.pow(src_rgb[:,1],2), torch.pow(src_rgb[:,2],2),src_rgb[:,0]* src_rgb[:,1],src_rgb[:,0]* src_rgb[:,2],src_rgb[:,1]* src_rgb[:,2]]), 0)
        elif self.extend_mode == 2:
            src_rgb = torch.stack(([src_rgb[:,0], src_rgb[:,1], src_rgb[:,2], torch.pow(src_rgb[:,0],2), torch.pow(src_rgb[:,1],2), torch.pow(src_rgb[:,2],2),src_rgb[:,0]* src_rgb[:,1],src_rgb[:,0]* src_rgb[:,2],src_rgb[:,1]* src_rgb[:,2], torch.pow(src_rgb[:,0],3), torch.pow(src_rgb[:,1],3), torch.pow(src_rgb[:,2],3)]), 0)



        return src_rgb 
 
    def prepare(self):
        self.TransM = np.dot(self.M.T, self.B)
        return
    
    def load(self, setup_file):
        weight = np.load(setup_file)
        
        self.M = weight["a"]
        self.B = weight["b"]
        self.C = weight["c"]
        
#         self.TransMin = weight['d'][0]
#         self.TransMax = weight['d'][1]
        
        self.extend_mode = weight['e'][0]
        #print(self.extend_mode)
        self.prepare()
        
        return
 
    def transfer1D(self, src_data):
        
        src_data = src_data.astype(np.float32)

        tar_sepc = np.dot(self.get_extend(src_data).T, self.TransM) + self.C
        
        tar_sepc = (tar_sepc - self.TransMin) / (self.TransMax - self.TransMin)

        return tar_sepc
    
    def transfer(self, src_data):
        
        src_data = src_data.astype(np.float32)
        
        src_shape = src_data.shape

        if len(src_shape) >= 3:
            src_data = src_data.reshape(-1, src_shape[-1])
            
        tar_sepc = np.dot(self.get_extend(src_data).T, self.TransM) + self.C

        if len(src_shape) >= 3:
            tar_sepc = tar_sepc.reshape(src_shape[0:-1] + tuple([self.C.shape[0]]))

        tar_sepc = (tar_sepc - self.TransMin) / (self.TransMax - self.TransMin)
        
        return tar_sepc
    
    def transfer_select(self, src_data,interval,band):
    
        src_data = src_data.astype(np.float32)
        
        src_shape = src_data.shape
        # print("src shape",src_shape)
        TransM = self.TransM[:,::interval]
        C = self.C[::interval]
        TransM_band = TransM[:,band]
        # print(TransM.shape)
        # print(C.shape)
        C_band = C[band]
        #print(C_band)
        # print("src_data shape1",src_data.shape)
        if len(src_shape) >= 3:
            src_data = src_data.reshape(-1, src_shape[-1])
        # print("src_data shape2",src_data.shape)
        #print('1' ,self.get_extend(src_data).T.shape)
        # print(self.get_extend(src_data).shape)
        tar_sepc = np.dot(self.get_extend(src_data).T, TransM_band) + C_band
        # print("tar_sepc shape2",tar_sepc.shape)
        if len(src_shape) >= 3:
            tar_sepc = tar_sepc.reshape(src_shape[0:-1] + tuple([C_band.shape[0]]))
        # print("tar_sepc shape2",tar_sepc.shape)
        tar_sepc = (tar_sepc - self.TransMin) / (self.TransMax - self.TransMin)
        # print("tar_sepc shape2",tar_sepc.shape)
        return tar_sepc
    
    def transfer_batch(self, src_data):
        src_data = src_data.astype(np.float32)
        src_shape = src_data.shape

        if len(src_shape) >= 4:
            src_data = src_data.reshape(src_data,[src_shape[0], -1, src_shape[-1]])

        tar_sepc = np.dot(self.get_extend(src_data).permute(0, 2, 3, 1), self.TransM) + self.C

        if len(src_shape) >= 4:
            tar_sepc = tar_sepc.reshape(src_shape[:-1] + (self.C.shape[0],))

        tar_sepc = (tar_sepc - self.TransMin) / (self.TransMax - self.TransMin)
    def transfer_select_batch(self, src_data, interval, band):
        # src_data = src_data.astype(np.float32)
        # src_shape = src_data.shape

        if torch.cuda.is_available():
            device = 'cuda:0'
        device = torch.device(device)
        src_data = src_data.permute(0,2,3,1)
        batch_data = src_data.type(torch.float32)
        batch_shape = batch_data.shape
        TransM = np.float32(self.TransM[:, ::interval])
        C = np.float32(self.C[::interval])
        TransM_band = TransM[:, band]
        C_band = C[band]

        TransM_band = torch.tensor(TransM_band)
        C_band = torch.tensor(C_band)
        TransM_band = TransM_band.to(device)
        C_band = C_band.to(device)

        if len(batch_shape) >= 4:
            batch_data = batch_data.reshape(-1, batch_shape[-2] * batch_shape[-3], batch_shape[-1])
        tar_sepc = torch.matmul(self.get_extend_batch(batch_data).permute(1,2,0), TransM_band) + C_band
        if len(batch_shape) >= 4:
            tar_sepc = tar_sepc.reshape(-1, batch_shape[-2], batch_shape[-3], 11)
        tar_sepc = (tar_sepc - self.TransMin) / (self.TransMax - self.TransMin)
        # tar_sepc = torch.tensor(tar_sepc)
        tar_sepc = tar_sepc.permute(0,3,1,2)
        return tar_sepc