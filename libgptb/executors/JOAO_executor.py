import os
import time
import json
import torch
import datetime
import numpy as np

import libgptb.losses as L
import libgptb.augmentors as A
import torch.nn.functional as F
from logging import getLogger
from torch.utils.tensorboard import SummaryWriter
from libgptb.executors.abstract_executor import AbstractExecutor
from libgptb.utils import ensure_dir
from functools import partial
from libgptb.evaluators import get_split,SVMEvaluator,RocAucEvaluator,PyTorchEvaluator,Logits_GraphCL,APEvaluator,MLPRegressionModel
from libgptb.models import DualBranchContrast
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error


class JOAOExecutor(AbstractExecutor):
    def __init__(self, config, model, data_feature):
        self.config=config
        self.data_feature=data_feature
        self.device = self.config.get('device', torch.device('cpu'))
        self.model = model.to(self.device)
        self.exp_id = self.config.get('exp_id', None)

        self.cache_dir = './libgptb/cache/{}/model_cache'.format(self.exp_id)
        self.evaluate_res_dir = './libgptb/cache/{}/evaluate_cache'.format(self.exp_id)
        self.summary_writer_dir = './libgptb/cache/{}/'.format(self.exp_id)
        ensure_dir(self.cache_dir)
        ensure_dir(self.evaluate_res_dir)
        ensure_dir(self.summary_writer_dir)

        self._writer = SummaryWriter(self.summary_writer_dir)
        self._logger = getLogger()
        self._scaler = self.data_feature.get('scaler')
        self._logger.info(self.model)
        for name, param in self.model.named_parameters():
            self._logger.info(str(name) + '\t' + str(param.shape) + '\t' +
                              str(param.device) + '\t' + str(param.requires_grad))
        total_num = sum([param.nelement() for param in self.model.parameters()])
        self._logger.info('Total parameter numbers: {}'.format(total_num))
        self.log_interval=self.config.get("log_interval",10)

        self.epochs=self.config.get("epochs",100)
        self.batch_size=self.config.get("batch_size",128)
        self.learning_rate = self.config.get('learning_rate', 0.001)
        self.learner = self.config.get('learner', 'adam')
        self.weight_decay = self.config.get('weight_decay', 0)
        self.lr_decay = self.config.get('lr_decay', True)
        self.lr_scheduler_type = self.config.get('lr_scheduler', 'multisteplr')
        self.lr_decay_ratio = self.config.get('lr_decay_ratio', 0.1)
        self.milestones = self.config.get('steps', [])
        self.step_size = self.config.get('step_size', 10)
        self.lr_lambda = self.config.get('lr_lambda', lambda x: x)
        self.lr_T_max = self.config.get('lr_T_max', 30)
        self.lr_eta_min = self.config.get('lr_eta_min', 0)
        self.lr_patience = self.config.get('lr_patience', 10)
        self.lr_threshold = self.config.get('lr_threshold', 1e-4)
        self.clip_grad_norm = self.config.get('clip_grad_norm', False)
        self.max_grad_norm = self.config.get('max_grad_norm', 1.)
        self.use_early_stop = self.config.get('use_early_stop', False)
        self.patience = self.config.get('patience', 50)
        self.log_every = self.config.get('log_every', 1)
        self.saved = self.config.get('saved_model', True)
        self.load_best_epoch = self.config.get('load_best_epoch', False)
        self.hyper_tune = self.config.get('hyper_tune', False)


        self.aug=self.config.get("augs","dnodes")
        self.dataset=self.config.get('dataset')
        self.local=self.config.get("local")=="True"
        self.prior=self.config.get("prior")=="True"
        self.DS=self.config.get("DS","MUTAG")
        self.hidden_dim = self.config.get('hidden_dim')
        self.num_layers = self.config.get('num_layers')
        self.num_class = self.config.get('num_class')
        self.label_dim = data_feature.get('label_dim')
        self.downstream_task=config.get("downstream_task","original")
        self.train_ratio = self.config.get("train_ratio",0.8)
        self.valid_ratio = self.config.get("valid_ratio",0.1)
        self.test_ratio = self.config.get("test_ratio",0.1)
        self.downstream_ratio=self.config.get("downstream_ratio",0.1)
        self.mode=self.config.get("mode","fast")
        self.gamma=config.get("gamma",0.01)
        self.num_samples = self.data_feature.get('num_samples')

        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        self.lr_scheduler = self._build_lr_scheduler()
        self._epoch_num = self.config.get('epoch', 0)
        if self._epoch_num > 0:
            self.load_model_with_epoch(self._epoch_num)
        self.loss_func = None

    def save_model(self, cache_name):
        """
        save model to cache_name

        Args:
            cache_name(str): name to save as
        """
        ensure_dir(self.cache_dir)
        self._logger.info("Saved model at " + cache_name)
        torch.save((self.model.state_dict(), self.optimizer.state_dict()), cache_name)

    def load_model(self, cache_name):
        """
        load model from cache_name

        Args:
            cache_name(str): name to load from
        """
        self._logger.info("Loaded model at " + cache_name)
        model_state, optimizer_state = torch.load(cache_name)
        self.model.load_state_dict(model_state)
        self.optimizer.load_state_dict(optimizer_state)
    
    def save_model_with_epoch(self, epoch):

        ensure_dir(self.cache_dir)
        config = dict()
        config['model_state_dict'] = self.model.state_dict()
        config['optimizer_state_dict'] = self.optimizer.state_dict()
        config['epoch'] = epoch
        model_path = self.cache_dir + '/' + self.config['model'] + '_' + self.config['dataset'] + '_epoch%d.tar' % epoch
        torch.save(config, model_path)
        self._logger.info("Saved model at {}".format(epoch))
        return model_path

    def load_model_with_epoch(self, epoch):
        """
        load model of the given epoch

        Args:
            epoch(int): 轮数
        """
        model_path = self.cache_dir + '/' + self.config['model'] + '_' + self.config['dataset'] + '_epoch%d.tar' % epoch
        assert os.path.exists(model_path), 'Weights at epoch %d not found' % epoch
        checkpoint = torch.load(model_path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self._logger.info("Loaded model at {}".format(epoch))

    def _build_optimizer(self):
        """
        chose 'optimizer' according to 'learner'
        """
        self._logger.info('You select `{}` optimizer.'.format(self.learner.lower()))
        if self.learner.lower() == 'adam':
            optimizer = torch.optim.Adam(self.model.encoder_model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'sgd':
            optimizer = torch.optim.SGD(self.model.encoder_model.parameters(), lr=self.learning_rate,
                                        momentum=self.lr_momentum, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'adagrad':
            optimizer = torch.optim.Adagrad(self.model.encoder_model.parameters(), lr=self.learning_rate,
                                            eps=self.lr_epsilon, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'rmsprop':
            optimizer = torch.optim.RMSprop(self.model.encoder_model.parameters(), lr=self.learning_rate,
                                            alpha=self.lr_alpha, eps=self.lr_epsilon,
                                            momentum=self.lr_momentum, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'sparse_adam':
            optimizer = torch.optim.SparseAdam(self.model.encoder_model.parameters(), lr=self.learning_rate,
                                               eps=self.lr_epsilon, betas=self.lr_betas)
        else:
            self._logger.warning('Received unrecognized optimizer, set default Adam optimizer')
            optimizer = torch.optim.Adam(self.model.encoder_model.parameters(), lr=self.learning_rate,
                                         eps=self.lr_epsilon, weight_decay=self.weight_decay)
        return optimizer

    def _build_lr_scheduler(self):
        """
        chose 'lr_scheduler' according to 'lr_scheduler'
        """
        if self.lr_decay:
            self._logger.info('You select `{}` lr_scheduler.'.format(self.lr_scheduler_type.lower()))
            if self.lr_scheduler_type.lower() == 'multisteplr':
                lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    self.optimizer, milestones=self.milestones, gamma=self.lr_decay_ratio)
            elif self.lr_scheduler_type.lower() == 'steplr':
                lr_scheduler = torch.optim.lr_scheduler.StepLR(
                    self.optimizer, step_size=self.step_size, gamma=self.lr_decay_ratio)
            elif self.lr_scheduler_type.lower() == 'exponentiallr':
                lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                    self.optimizer, gamma=self.lr_decay_ratio)
            elif self.lr_scheduler_type.lower() == 'cosineannealinglr':
                lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer, T_max=self.lr_T_max, eta_min=self.lr_eta_min)
            elif self.lr_scheduler_type.lower() == 'lambdalr':
                lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                    self.optimizer, lr_lambda=self.lr_lambda)
            elif self.lr_scheduler_type.lower() == 'reducelronplateau':
                lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer, mode='min', patience=self.lr_patience,
                    factor=self.lr_decay_ratio, threshold=self.lr_threshold)
            else:
                self._logger.warning('Received unrecognized lr_scheduler, '
                                     'please check the parameter `lr_scheduler`.')
                lr_scheduler = None
        else:
            lr_scheduler = None
        return lr_scheduler

    def compute_metrics(self, predictions, targets):
        predictions = predictions.cpu().numpy()
        targets = targets.cpu().numpy()
        mse = mean_squared_error(targets, predictions)
        rmse = mse ** 0.5
        mae = mean_absolute_error(targets, predictions)
        mape = mean_absolute_percentage_error(targets, predictions)
        return rmse, mae, mape

    def downstream_regressor(self,dataloader):
         
        input_dim = self.hidden_dim*self.num_layers   
        nhid = 128   
        output_dim = 1    

        regressor = None
        optimizer = None
        criterion = torch.nn.MSELoss()

         
        downstream_ratio = self.downstream_ratio   
        test_ratio = 0.1   

       
        num_samples = len(dataloader['full'])
        print(f'num_samples is {num_samples}')
        num_train = int(num_samples * downstream_ratio)
        print(f'num_train is {num_train}')
        num_test = int(num_samples * (1-test_ratio))
        print(f'num_test is {num_test}')

        num_epochs = 20
        best_test_rmse = float('inf')
        best_test_mae = float('inf')
        best_test_mape = float('inf')
        
        regressor = MLPRegressionModel(input_dim, nhid, output_dim).to(self.device)
        optimizer = torch.optim.Adam(regressor.parameters(), lr=0.001)

        for epoch in range(num_epochs):
            train_loss = 0
            test_loss = 0
            correct = 0
            
            for i, batch_g in enumerate(dataloader['full']):
                data = batch_g.to(self.device)
                feat = data.x
                labels = data.y  
                _, out, _, _, _, _ = self.model.encoder_model(data.x, data.edge_index, data.batch)
                
                if i < num_train:
                    regressor.train()
                    optimizer.zero_grad()
                    output = regressor(out)
                    loss = criterion(output, labels.to(self.device).unsqueeze(1))  
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                     
                else:
                    print(f'i is {i}')
                    break
            self._logger.info(f'Downstream Epoch: {epoch+1}, Training Loss: {train_loss:.4f}')
            with torch.no_grad():
                regressor.eval()
                all_predictions = []
                all_labels = []
                for j, test_batch in enumerate(dataloader['full']):
                    
                    if j >= num_test:
                        self._logger.debug(f'Processing batch: {j}')
                        test_batch = test_batch.to(self.device)
                        self._logger.debug('Batch moved to device')
                        _,test_out, _, _, _, _ = self.model.encoder_model(test_batch.x, test_batch.edge_index, test_batch.batch)
                        self._logger.debug(f'Encoder model output: {test_out}')
                        test_output = regressor(test_out)
                        self._logger.debug(f'Regressor output: {test_output}')
                        all_predictions.append(test_output.cpu())
                        self._logger.debug(f'Predictions appended: {test_output.cpu()}')
                        all_labels.append(test_batch.y.cpu().float().unsqueeze(1))
                        self._logger.debug(f'Labels appended: {test_batch.y.cpu().float().unsqueeze(1)}')
                    
                
                all_predictions = torch.cat(all_predictions, dim=0)
                all_labels = torch.cat(all_labels, dim=0)
                rmse, mae, mape = self.compute_metrics(all_predictions, all_labels)

                if mae < best_test_mae:
                    best_test_rmse = rmse
                    best_test_mae = mae
                    best_test_mape = mape
                    
            print(f'Epoch: {epoch+1}, Training Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test MAE: {mae:.4f}')
            
        result = {
        'best_test_rmse': float(best_test_rmse),
        'best_test_mae': float(best_test_mae),
        'best_test_mape': float(best_test_mape)
        }
        
        filename = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S') + '_' + \
                    self.config['model'] + '_' + self.config['dataset']
        save_path = self.evaluate_res_dir
        with open(os.path.join(save_path, '{}.json'.format(filename)), 'w') as f:
            json.dump(result, f)
            self._logger.info('Evaluate result is saved at ' + os.path.join(save_path, '{}.json'.format(filename)))
        return result
    
  



    def evaluate(self, dataloader):
        """
        use model to test data

        Args:
            test_dataloader(torch.Dataloader): Dataloader
        """
        self._logger.info('Start evaluating ...')
         
        for epoch_idx in [100-1]:
            self.load_model_with_epoch(epoch_idx)
            if self.downstream_task in ['original','both']:
                if self.config['dataset'] in ['PCQM4Mv2','ZINC_full']:
                    self.model.encoder_model.eval()
                    result=self.downstream_regressor(dataloader)
                    self._logger.info(f'(E): Best test RMSE={result["best_test_rmse"]:.4f}, MAE={result["best_test_mae"]:.4f}, MAPE={result["best_test_mape"]:.4f}')
                    
                else:
                    self.model.encoder_model.eval()
                    x = []
                    y = []
                    for data in dataloader['full']:
                        data = data.to('cuda')
                        if data.x is None:
                            num_nodes = data.batch.size(0)
                            data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)
                        with torch.no_grad():
                            _, g, _, _, _, _ = self.model.encoder_model(data.x, data.edge_index, data.batch)
                            x.append(g)
                            y.append(data.y)
                        torch.cuda.empty_cache()
                    x = torch.cat(x, dim=0)
                    y = torch.cat(y, dim=0)

                    split = get_split(num_samples=self.num_samples, train_ratio=0.8, test_ratio=0.1,downstream_ratio = self.downstream_ratio, dataset=self.config['dataset'])
                    if self.config['dataset'] == 'ogbg-molhiv': 
                        result = RocAucEvaluator()(x, y, split)
                        print(f'(E): Roc-Auc={result["roc_auc"]:.4f}')
                    elif self.config['dataset'] == 'ogbg-ppa':
                         
                        self._logger.info('nclasses is {}'.format(self.num_class))
                        result = PyTorchEvaluator(n_features=x.shape[1],n_classes=self.num_class)(x, y, split)
                    elif self.config['dataset'] == 'ogbg-molpcba':
                        result = APEvaluator(self.hidden_dim*self.num_layers, self.label_dim)(x, y, split)
                        self._logger.info(f'(E): ap={result["ap"]:.4f}')
                     
                    else:
                        result = SVMEvaluator()(x, y, split)
                        print(f'(E): Best test F1Mi={result["micro_f1"]:.4f}, F1Ma={result["macro_f1"]:.4f}')
                    self._logger.info('Evaluate result is ' + json.dumps(result))
                    
                if self.downstream_task == 'loss' or self.downstream_task == 'both':
                    losses = self._train_epoch(dataloader["test"],epoch_idx, self.loss_func,train = False)
                    result = np.mean(losses) 
                    self._logger.info('Evaluate loss is ' + json.dumps(result))
                    
                
                filename = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S') + '_' + \
                            self.config['model'] + '_' + self.config['dataset']
                save_path = self.evaluate_res_dir
                with open(os.path.join(save_path, '{}.json'.format(filename)), 'w') as f:
                    json.dump(result, f)
                    self._logger.info('Evaluate result is saved at ' + os.path.join(save_path, '{}.json'.format(filename)))
        
    def train(self, train_dataloader, eval_dataloader):
        self._logger.info('Start training ...')
        min_val_loss = float('inf')
        wait = 0
        best_epoch = 0
        train_time = []
        eval_time = []
        num_batches = len(train_dataloader)
        self._logger.info("num_batches:{}".format(num_batches))

        for epoch_idx in range(self._epoch_num, self.epochs):
            start_time = time.time()
            losses = self._train_epoch(train_dataloader, epoch_idx, self.loss_func)
            t1 = time.time()
            train_time.append(t1 - start_time)
            self._writer.add_scalar('training loss', np.mean(losses), epoch_idx)
            self._logger.info("epoch complete!")

            self._logger.info("evaluating now!")
            t2 = time.time()
            val_loss = np.mean(losses)
            end_time = time.time()
            eval_time.append(end_time - t2)

            if self.lr_scheduler is not None:
                if self.lr_scheduler_type.lower() == 'reducelronplateau':
                    self.lr_scheduler.step(val_loss)
                else:
                    self.lr_scheduler.step()

            if (epoch_idx % self.log_every) == 0:
                log_lr = self.optimizer.param_groups[0]['lr']
                message = 'Epoch [{}/{}] train_loss: {:.4f}, lr: {:.6f}, {:.2f}s'.\
                    format(epoch_idx, self.epochs, np.mean(losses),  log_lr, (end_time - start_time))
                self._logger.info(message)

            
            if epoch_idx+1 in range(5,101,5):
                model_file_name = self.save_model_with_epoch(epoch_idx)
                self._logger.info('saving to {}'.format(model_file_name))
            
            if val_loss < min_val_loss:
                wait = 0
                if self.saved:
                    model_file_name = self.save_model_with_epoch(epoch_idx)
                    self._logger.info('Val loss decrease from {:.4f} to {:.4f}, '
                                      'saving to {}'.format(min_val_loss, val_loss, model_file_name))
                min_val_loss = val_loss
                best_epoch = epoch_idx
            else:
                wait += 1
                if wait == self.patience and self.use_early_stop:
                    self._logger.warning('Early stopping at epoch: %d' % epoch_idx)
                    break
        if len(train_time) > 0:
            self._logger.info('Trained totally {} epochs, average train time is {:.3f}s, '
                              'average eval time is {:.3f}s'.
                              format(len(train_time), sum(train_time) / len(train_time),
                                     sum(eval_time) / len(eval_time)))
        if self.load_best_epoch:
            self.load_model_with_epoch(best_epoch)
        return min_val_loss


    def _train_epoch(self, train_dataloader,epoch_idx,loss_func,train=True):
        if train:
            self.model.encoder_model.train()
        else:
            self.model.encoder_model.eval()
        epoch_loss = 0
        for data in train_dataloader:
            data = data.to(self.device)
            if train:
                self.optimizer.zero_grad()

            if data.x is None:
                num_nodes = data.batch.size(0)
                data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)

            
            z, g, z1, z2, g1, g2 = self.model.encoder_model(data.x, data.edge_index, data.batch)
            g1, g2 = [self.model.encoder_model.encoder.project(g) for g in [g1, g2]]
            z1, z2 = [self.model.encoder_model.encoder.project(z) for z in [z1, z2]]
            loss = self.model.contrast_model(g1=g1, g2=g2,h1=z1, h2=z2, batch=data.batch)
            if train:
                loss.backward()
                self.optimizer.step()
            epoch_loss += loss.item()

        #minimax 
        loss_aug = np.zeros(4)
        for n in range(4):
            _aug_P = np.zeros(4)
            _aug_P[n] = 1
            dataset_aug_P = _aug_P
            count, count_stop = 0, len(train_dataloader)//4+1
            with torch.no_grad():
                 for data in train_dataloader:
                    data = data.to(self.device)
                    self.optimizer.zero_grad()

                    if data.x is None:
                        num_nodes = data.batch.size(0)
                        data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)

                     
                    z, g, z1, z2, g1, g2 = self.model.encoder_model(data.x, data.edge_index, data.batch)
                    g1, g2 = [self.model.encoder_model.encoder.project(g) for g in [g1, g2]]
                    z1, z2 = [self.model.encoder_model.encoder.project(z) for z in [z1, z2]]
                    loss = self.model.contrast_model(g1=g1, g2=g2,h1=z1, h2=z2, batch=data.batch)
                    loss_aug[n] += loss.item()*data.num_graphs
                    if self.mode == 'fast':
                            count += 1
                            if count == count_stop:
                                break
            if self.mode == 'fast':
                loss_aug[n] /= (count_stop*self.batch_size)
            else:
                loss_aug[n] /= len(train_dataloader.dataset)

        gamma = float(self.gamma)
        beta = 1
        b = self.model.aug_P + beta * (loss_aug - gamma * (self.model.aug_P - 1/5))

        mu_min, mu_max = b.min()-1/4, b.max()-1/4
        mu = (mu_min + mu_max) / 2
        # bisection method
        while abs(np.maximum(b-mu, 0).sum() - 1) > 1e-2:
            if np.maximum(b-mu, 0).sum() > 1:
                mu_min = mu
            else:
                mu_max = mu
            mu = (mu_min + mu_max) / 2

        self.model.aug_P = np.maximum(b-mu, 0)
        self.model.aug_P /= np.sum(self.model.aug_P)
        self.model._update_aug2()
        
        return epoch_loss
        