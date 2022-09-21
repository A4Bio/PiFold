from tqdm import tqdm
import numpy as np
import torch.nn as nn
import torch

from .base_method import Base_method
from .utils import cuda
from .prodesign_model import ProDesign_Model

class ProDesign(Base_method):
    def __init__(self, args, device, steps_per_epoch):
        Base_method.__init__(self, args, device, steps_per_epoch)
        self.model = self._build_model()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer, self.scheduler = self._init_optimizer(steps_per_epoch)

    def _build_model(self):
        return ProDesign_Model(self.args).to(self.device)

    def test_one_epoch(self, test_loader):
        self.model.eval()
        test_sum, test_weights = 0., 0.
        test_pbar = tqdm(test_loader)

        with torch.no_grad():
            for batch in test_pbar:
                X, S, score, mask, lengths = cuda(batch, device = self.device)
                X, S, score, h_V, h_E, E_idx, batch_id, mask_bw, mask_fw, decoding_order = self.model._get_features(S, score, X=X, mask=mask)
                log_probs = self.model(h_V, h_E, E_idx, batch_id)
                loss, loss_av = self.loss_nll_flatten(S, log_probs)
                mask = torch.ones_like(loss)
                test_sum += torch.sum(loss * mask).cpu().data.numpy()
                test_weights += torch.sum(mask).cpu().data.numpy()
                test_pbar.set_description('test loss: {:.4f}'.format(loss.mean().item()))
            
            test_recovery, test_subcat_recovery = self._cal_recovery(test_loader.dataset, test_loader.featurizer)
            
        test_loss = test_sum / test_weights
        test_perplexity = np.exp(test_loss)
    
        return test_perplexity, test_recovery, test_subcat_recovery

    def _cal_recovery(self, dataset, featurizer):
        recovery = []
        subcat_recovery = {}
        with torch.no_grad():
            for protein in tqdm(dataset):
                p_category = protein['category'] if 'category' in protein.keys() else 'Unknown'
                if p_category not in subcat_recovery.keys():
                    subcat_recovery[p_category] = []

                protein = featurizer([protein])
                X, S, score, mask, lengths = cuda(protein, device = self.device)
                X, S, score, h_V, h_E, E_idx, batch_id, mask_bw, mask_fw, decoding_order = self.model._get_features(S, score, X=X, mask=mask)
                log_probs = self.model(h_V, h_E, E_idx, batch_id)
                S_pred = torch.argmax(log_probs, dim=1)
                cmp = (S_pred == S)
                recovery_ = cmp.float().mean().cpu().numpy()

                if np.isnan(recovery_): recovery_ = 0.0

                subcat_recovery[p_category].append(recovery_)
                recovery.append(recovery_)
            
            for key in subcat_recovery.keys():
                subcat_recovery[key] = np.median(subcat_recovery[key])

        self.mean_recovery = np.mean(recovery)
        self.std_recovery = np.std(recovery)
        self.min_recovery = np.min(recovery)
        self.max_recovery = np.max(recovery)
        self.median_recovery = np.median(recovery)
        recovery = np.median(recovery)
        return recovery, subcat_recovery

    def loss_nll_flatten(self, S, log_probs):
        """ Negative log probabilities """
        criterion = torch.nn.NLLLoss(reduction='none')
        loss = criterion(log_probs, S)
        loss_av = loss.mean()
        return loss, loss_av
    
    def loss_nll_smoothed(self, S, log_probs, weight=0.1):
        """ Negative log probabilities """
        S_onehot = torch.nn.functional.one_hot(S, num_classes=20).float()
        S_onehot = S_onehot + weight / float(S_onehot.size(-1))
        S_onehot = S_onehot / S_onehot.sum(-1, keepdim=True) # [4, 463, 20]/[4, 463, 1] --> [4, 463, 20]

        loss = -(S_onehot * log_probs).sum(-1).mean()
        loss_av = torch.sum(loss)
        return loss, loss_av