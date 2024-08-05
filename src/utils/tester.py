from .trainer import Trainer
import numpy as np
import os.path as P
import torch


class Tester(object):
    """A functional class facilitating and supporting all procedures in training phase, based on Trainer class"""

    def __init__(self, model_cube, data_cube, criterion_cube, writer_cube,
                 lr_scheme, snapshot_scheme, device, wrap_test=True):
        # wrap_test = True: don't load pretrain / resume weights
        self._trainer = Trainer(model_cube, data_cube, criterion_cube, writer_cube,
                                lr_scheme, snapshot_scheme, device, True)

    def get_model(self):
        return self._trainer.model

    def get_train_loader(self):
        return self._trainer.trainseqloader

    def test(self, state_suffix, save_dir, is_indiv=False, is_save_nii=False,
             is_cc=False, is_true_test=False):
        self._trainer.test(state_suffix, save_dir, is_indiv, is_save_nii,
                           is_cc, is_true_test)

    def test_given_pretrain(self, pretrain, save_dir, is_indiv=False, is_save_nii=False,
             is_cc=False, is_true_test=False):
        self._trainer.test_given_pretrain(pretrain, save_dir, is_indiv, is_save_nii,
                           is_cc, is_true_test)

    def test_as_is(self, folder='results', is_save_nii=False):
        self._trainer.model.to(self._trainer.device)
        self._trainer.test(None, folder, True, is_save_nii,
                           False, False)

    def snapshot(self, fname=None, compress=False):
        state_dict = {
            'state_dict': self._trainer.model.state_dict(),
        }
        if fname:
            filename = P.join(self._trainer.root, fname)
        else:
            filename = P.join(self._trainer.root, 'state_ptq.pkl')
        print(f'Snapshotting to {filename}')
        if compress:
            for k, v in state_dict['state_dict'].items():
                state_dict['state_dict'][k] = v.data.cpu().numpy()
            np.savez_compressed(filename, state_dict)
        else:
            torch.save(state_dict, filename)

    def perform_quantization(self):
        if self._trainer._final_quantization():
            self._trainer._final_snap('finalQ')
            print('Quantized and saved to finalQ.')
        else:
            print('No quantization is available.')


class PTQTester(Tester):
    """A functional class that is used for PTQ with simplified init arguments"""
    def __init__(self, model_cube, data_cube, snapshot_scheme, device):
        super().__init__(model_cube, data_cube, {}, {},
                         {}, snapshot_scheme, device, wrap_test=True)
