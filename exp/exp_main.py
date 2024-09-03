from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import Transformer, Informer, Autoformer, DLinear, Linear, NLinear, FEDformer
from ns_models import ns_Transformer, ns_Informer, ns_Autoformer
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
from torchsummary import summary

import numpy as np
import torch
import torch.nn as nn
from torch import optim
import prettytable
from prettytable import PrettyTable
import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings('ignore')

import sparselearning
from sparselearning.core import Masking, CosineDecay, LinearDecay

import pickle
def save_obj(name, obj):
    with open( name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open( name + '.pkl', 'rb') as f:
        return pickle.load(f)

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: 
            continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params
    
    
class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'Transformer': Transformer,
            'Informer': Informer,
            'Autoformer': Autoformer,
            'ns_Transformer': ns_Transformer,
            'ns_Informer': ns_Informer,
            'ns_Autoformer': ns_Autoformer,
            'DLinear': DLinear,
            'NLinear': NLinear,
            'Linear': Linear,
            'FEDformer': FEDformer,
        }
        model = model_dict[self.args.model].Model(self.args).float()
        #summary(model)
        #print(summary(model))
        #print(model)
        
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting, args=None):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = self.args.path + self.args.checkpoints
        print(path)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        
        ############################################################
        mask = None
        print("param counts = ", count_parameters(self.model))
        if args.sparse:
            decay = CosineDecay(args.prune_rate, len(train_loader)*(args.train_epochs*args.multiplier))
            #decay = LinearDecay(args.prune_rate, frequency = 20)
            mask = Masking(model_optim, prune_rate=args.prune_rate, death_mode=args.prune, prune_rate_decay=decay, growth_mode=args.growth,
                           redistribution_mode=args.redistribution, args=args, train_loader=train_loader,
                           dense_params =count_parameters(self.model))
            mask.add_module(self.model, sparse_init=args.sparse_init)

        ############################################################
        
        
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
        
        best_loss = 10000
        status = "prune"
        last_status = "prune"
        best_sparsity = 1 - args.init_density
        best_vali_loss = 10000
        best_test_loss = 10000

        for epoch in range(self.args.train_epochs):
            print("\n\n\n###########################################################################################")
            print("###############                             EPOCH ", epoch, "                         ##############")
            print("###########################################################################################")
            
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if 'Linear' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
         
                    if self.args.sparse and (mask.steps+1) % mask.prune_every_k_steps == 0:
                        
                        ######### update status for PALS
                        if  self.args.method == 'PALS':
                            last_status = status
                            val_loss = self.vali(vali_data, vali_loader, criterion)
                            if mask.sparsity_level < 0.2:
                                status = "prune"
                            elif val_loss <= args.loss_freedom_factor * best_loss and mask.sparsity_level < 0.90:
                                status = "prune"
                            elif val_loss > args.loss_freedom_factor * best_loss  and mask.sparsity_level > best_sparsity: 
                                status = "grow"
                            else:
                                status = "stable"
                            if val_loss < best_loss:
                                best_loss = val_loss
                                best_sparsity = mask.sparsity_level
                            print("================================ best_sparsity =====> ", best_sparsity)
                            print("================================ status =====> ", status , " mask.steps", mask.steps)
      
                    if mask is not None: mask.step(status)
                    else: model_optim.step()
            
            if mask is not None:
                mask.print_nonzero_counts()
                
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)
            if vali_loss < best_vali_loss:
                best_vali_loss = vali_loss
                best_test_loss = test_loss
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            
            ########### Save mask when saving the checkpoint if the model is not dense
            mask_to_save = None
            if mask is not None:
                print("mask.sparsity_level  = ", mask.sparsity_level , "  self.args.final_density: ", self.args.final_density)
                mask_to_save = mask     
            
            ########## Early-stopping or not?
            if mask is not None and self.args.method not in ['PALS']:
                cond = mask.sparsity_level >= (1- self.args.final_density - 0.01)     
            else:
                cond = True
            if cond:
                early_stopping(vali_loss, self.model, path, mask = mask_to_save)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break
                    
            
            ######## Adjust the learning rate
            adjust_learning_rate(model_optim, epoch + 1, self.args)

        
        ######### Load the best model
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        ######### If model is sparse save sparsity
        if args.sparse:
            best_mask_path = path  + 'mask'
            best_mask = load_obj(best_mask_path)
        
        if args.sparse:
            os.remove(best_mask_path+".pkl")    
            os.remove(best_model_path)    
            return self.model, best_vali_loss, best_test_loss, best_mask
        else:
            os.remove(best_model_path)  
            return self.model

    def test(self, setting, test=0, mode=""):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join(self.args.path  + 'checkpoints/', 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path =self.args.path  + 'test_results/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    if mode != "":
                        visual(gt, pd, os.path.join(folder_path, mode + "_" + str(i) + '.pdf'))
                    else:
                        visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))
        
        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = self.args.path 
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        print('\nrmse:{}, mape:{}, mspe:{}'.format(rmse, mape, mspe))
        f = open(self.args.path +"result.txt", 'a')
        f.write(setting + "  \n")
        if mode != "":
            f.write("mode:"+ mode+"\n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\nrmse:{}, mape:{}, mspe:{}'.format(rmse, mape, mspe))
        f.write('\n')
        f.write('\n')
        f.close()

        return

    def predict(self, setting, mode="", load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.path , self.args.checkpoints )
            if not os.path.exists(path):
                os.makedirs(path)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = self.args.path 
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        if mode != "":
            np.save(folder_path + mode+'_real_prediction.npy', preds)
        else:
            np.save(folder_path + 'real_prediction.npy', preds)
        return