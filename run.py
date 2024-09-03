
import torch
from exp.exp_main import Exp_Main
from utils.arg_utils import parse_args
import random
import numpy as np
import sys

import os
import shutil
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
cudnn.benchmark = True
cudnn.deterministic = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import copy
import re


def main():
    
    args = parse_args()

    ############# Set Random seed
    fix_seed = args.seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    torch.random.manual_seed(fix_seed)
    #torch.use_deterministic_algorithms(True)
    np.random.seed(fix_seed)
    
    ############# GPU
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    print("args.use_gpu ", args.use_gpu, "  -  cuda: ", torch.cuda.is_available())
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    ############# Print arguments
    print('Args in experiment:')
    print(args)

    Exp = Exp_Main
    ############# Train model
    if args.is_training:
        ########################################################## Experiment identifier
        if args.features == "M":
            setting = '{}/{}/{}/{}/{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}/seed_{}'.format(
                args.data_path.replace('.csv',''),
                args.substring_sparse,
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.factor,
                args.embed,
                args.distil,
                args.des, args.seed)
        elif args.features == "S":
            setting = '{}_uni/{}/{}/{}/{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}/seed_{}'.format(
                args.data_path.replace('.csv',''),
                args.substring_sparse,
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.factor,
                args.embed,
                args.distil,
                args.des, args.seed)
        else:    
            setting = '{}/{}/{}/{}/{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}/seed_{}'.format(
                args.data_path.replace('.csv',''),
                args.substring_sparse,
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.factor,
                args.embed,
                args.distil,
                args.des, args.seed)
        ########################################################## Create results directory
        path = './results/' + setting + "/"
        args.path = path
        if os.path.exists(path):
            if os.path.exists(path+ "/result.txt"):
                print("\n\n\n\n\n\n\n")
                print("&&&&&&&&&&&&&&&&&&^^^^^^^^^^^^^^^^^^^^^^^^^^^&&&&&&&&&&&&&&&&&&&&&&&&&&")
                print("&&&&&&&&&&&&&&&&&&^^^^^^^^^^^^^^^^^^^^^^^^^^^&&&&&&&&&&&&&&&&&&&&&&&&&&")
                print("Path already exist")
                print("&&&&&&&&&&&&&&&&&&^^^^^^^^^^^^^^^^^^^^^^^^^^^&&&&&&&&&&&&&&&&&&&&&&&&&&")
                print("&&&&&&&&&&&&&&&&&&^^^^^^^^^^^^^^^^^^^^^^^^^^^&&&&&&&&&&&&&&&&&&&&&&&&&&\n\n\n\n\n")
                exit(0)
            else:
                shutil.rmtree(path)
        print("creating directory")
        os.makedirs(path) 
        log = open(path + "output.log", "a")
        sys.stdout = log
        
            
        
        
        
        
        if args.sparse and "PALS" in args.method:
            ############################################################  find prf,lff for seed=2020
            best_loss_exp, best_sparsity_exp = None, None
            f = open(args.path + "logs.txt", 'a')
            f.write(setting + "  \n")
        
            if args.seed == 2020:
                best_loss_model, best_loss_mask = None, None
                best_loss_valid_loss, best_loss_test_loss = 10000, 10000
                best_loss_prf, best_loss_lff = 0, 0

                best_sparsity_model, best_sparsity_mask = None, None
                best_sparsity_valid_loss, best_sparsity_test_loss = 10000, 10000
                best_sparsity_prf, best_sparsity_lff= 0, 0
                best_sparsity = 0
                ############# Train Model
                print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
                for prf in [1.05, 1.10, 1.20]:
                    for lff in [1.05, 1.1, 1.2]:
                        f.write("\n=================================================================")
                        args.prune_rate_factor = prf
                        args.loss_freedom_factor = lff 
                        f.write("\nprune_rate_factor= {}, loss_freedom_factor= {}".format(prf, lff))
                        exp = Exp(args)  # set experiments
                        model , valid_loss, test_loss, mask = exp.train(setting,  args=args)
                        f.write('\nsparsity level:{}, density level:{}'.format(mask.sparsity_level, 1 - mask.sparsity_level))
                        f.write("\nvalid_loss= {}, test_loss= {}".format(valid_loss, test_loss))
                        if valid_loss < best_loss_valid_loss:
                            best_loss_model, best_loss_mask = model, mask
                            best_loss_prf, best_loss_lff = prf, lff
                            best_loss_valid_loss = valid_loss
                            best_loss_test_loss = test_loss
                            best_loss_exp = copy.deepcopy(exp)
                        
                        if mask.sparsity_level > best_sparsity or (mask.sparsity_level==best_sparsity and valid_loss<best_sparsity_valid_loss):
                            best_sparsity_model, best_sparsity_mask = model, mask
                            best_sparsity_prf, best_sparsity_lff = prf, lff
                            best_sparsity_valid_loss = valid_loss
                            best_sparsity_test_loss = test_loss
                            best_sparsity = mask.sparsity_level
                            best_sparsity_exp = copy.deepcopy(exp)
                        del exp
                f.write("\n\n\nbest_loss prune_rate_factor= {}, best_loss loss_freedom_factor= {}".format(best_loss_prf, best_loss_lff))
                f.write("\nbestloss valid_loss= {}, bestloss test_loss= {}".format(best_loss_valid_loss, best_loss_test_loss))
                f.write("\nbestloss sparsity= {}".format(best_loss_mask.sparsity_level))
                f.write("\n\n\nbest_sparsity prune_rate_factor= {}, best_sparsity loss_freedom_factor= {}".format(best_sparsity_prf, best_sparsity_lff))
                f.write("\nbest_sparsity valid_loss= {}, best_sparsity test_loss= {}".format(best_sparsity_valid_loss, best_sparsity_test_loss))
                f.write("\nbest_sparsity sparsity= {}".format(best_sparsity_mask.sparsity_level))

            else:
                ############################################################ read prf/lff from seed 2020 and then train
                path_read_args = args.path.replace("seed_"+str(args.seed),'seed_2020')
                path_read_args = args.path.replace("results_sp",'results')
                with open(path_read_args +"logs.txt") as f2:
                    lines = f2.readlines()
                    for line in lines:
                        if "best_loss prune_rate_factor= " in line:
                            result = re.search('best_loss prune_rate_factor= (.*), best_loss', line)
                            best_loss_prf = float(result.group(1))
                            result = re.search('best_loss loss_freedom_factor= (.*)', line)
                            best_loss_lff = float(result.group(1))
                        elif "best_sparsity prune_rate_factor= " in line:
                            result = re.search('best_sparsity prune_rate_factor= (.*), best_sparsity', line)
                            best_sparsity_prf = float(result.group(1))
                            result = re.search('best_sparsity loss_freedom_factor= (.*)', line)
                            best_sparsity_lff = float(result.group(1))
                            
                f2.close()
                
                args.prune_rate_factor, args.loss_freedom_factor = best_loss_prf, best_loss_lff
                f.write("\n\n\nbest_loss prune_rate_factor= {}, best_loss loss_freedom_factor= {}".format(best_loss_prf, best_loss_lff))
                best_loss_exp = Exp(args)  # set experiments
                model, valid_loss, test_loss, mask = best_loss_exp.train(setting,  args=args)
                f.write("\nbestloss valid_loss= {}, bestloss test_loss= {}".format(valid_loss, test_loss))
                f.write("\nbestloss sparsity= {}".format(mask.sparsity_level))
                del mask, model
                
                args.prune_rate_factor, args.loss_freedom_factor = best_sparsity_prf, best_sparsity_lff
                f.write("\n\n\nbest_sparsity prune_rate_factor= {}, best_sparsity loss_freedom_factor= {}".format(best_sparsity_prf, best_sparsity_lff))
                best_sparsity_exp = Exp(args)  # set experiments\
                model, valid_loss, test_loss, mask = best_sparsity_exp.train(setting,  args=args)
                f.write("\nbest_sparsity valid_loss= {}, best_sparsity test_loss= {}".format(valid_loss, test_loss))
                f.write("\nbest_sparsity sparsity= {}".format(mask.sparsity_level))
                del mask, model
            f.close()
                
            ############# Test Model
            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            best_loss_exp.test(setting, mode = "loss")
            ############# Predict
            if args.do_predict:
                print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                best_loss_exp.predict(setting, mode = "loss")
                

            ############# Test Model
            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            best_sparsity_exp.test(setting, mode = "sparsity")
            ############# Predict
            if args.do_predict:
                print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                best_sparsity_exp.predict(setting, mode = "sparsity")
        

            
            
        else:
            exp = Exp(args)
            ############# Train Model
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting,  args=args)
            
            ############# Test Model
            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)
            
            ############# Predict
            if args.do_predict:
                print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.predict(setting, True)

        torch.cuda.empty_cache()
        
        
    else:
        ii = 0
        setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(args.model_id,
                                                                                                      args.model,
                                                                                                      args.data,
                                                                                                      args.features,
                                                                                                      args.seq_len,
                                                                                                      args.label_len,
                                                                                                      args.pred_len,
                                                                                                      args.d_model,
                                                                                                      args.n_heads,
                                                                                                      args.e_layers,
                                                                                                      args.d_layers,
                                                                                                      args.d_ff,
                                                                                                      args.factor,
                                                                                                      args.embed,
                                                                                                      args.distil,
                                                                                                      args.des, ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()




if __name__ == "__main__":
    main()

