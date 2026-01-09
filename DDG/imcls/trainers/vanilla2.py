import torch
from torch.nn import functional as F
# DDG-added
import torch.autograd.forward_ad as fwAD
from torch.func import functional_call

from dassl.engine import TRAINER_REGISTRY, TrainerX
# DDG-added
from dassl.engine import TrainerX_dfed
from dassl.metrics import compute_accuracy
from dassl.modeling.ops import random_mixstyle, crossdomain_mixstyle, sots_mixstyle

# DDG-added
from copy import deepcopy


@TRAINER_REGISTRY.register()
class Vanilla2(TrainerX):
    """Vanilla baseline.

    Slightly modified for mixstyle.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        mix = cfg.TRAINER.VANILLA2.MIX

        if mix == 'random':
            self.model.apply(random_mixstyle)
            print('MixStyle: random mixing')

        elif mix == 'crossdomain':
            self.model.apply(crossdomain_mixstyle)
            print('MixStyle: cross-domain mixing')

        # SOTS-added
        elif mix == 'sots':
            self.model.apply(sots)
            print('SOTS')

        else:
            raise NotImplementedError

    def forward_backward(self, batch):
        input, label = self.parse_batch_train(batch)
        output = self.model(input)
        loss = F.cross_entropy(output, label)
        self.model_backward_and_update(loss)

        loss_summary = {
            'loss': loss.item(),
            'acc': compute_accuracy(output, label)[0].item()
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch['img']
        label = batch['label']
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label
    
    @torch.no_grad()
    def vis(self):
        self.set_model_mode('eval')
        output_dir = self.cfg.OUTPUT_DIR
        source_domains = self.cfg.DATASET.SOURCE_DOMAINS
        print('Source domains:', source_domains)

        out_embed = []
        out_domain = []
        out_label = []

        split = self.cfg.TEST.SPLIT
        data_loader = self.val_loader if split == 'val' else self.test_loader

        print('Extracting style features')

        for batch_idx, batch in enumerate(data_loader):
            input = batch['img'].to(self.device)
            label = batch['label']
            domain = batch['domain']
            impath = batch['impath']

            # model should directly output features or style statistics
            raise NotImplementedError
            output = self.model(input)
            output = output.cpu().numpy()
            out_embed.append(output)
            out_domain.append(domain.numpy())
            out_label.append(label.numpy()) # CLASS LABEL

            print('processed batch-{}'.format(batch_idx + 1))

        out_embed = np.concatenate(out_embed, axis=0)
        out_domain = np.concatenate(out_domain, axis=0)
        out_label = np.concatenate(out_label, axis=0)
        print('shape of feature matrix:', out_embed.shape)
        out = {
            'embed': out_embed,
            'domain': out_domain,
            'dnames': source_domains,
            'label': out_label
        }
        out_path = osp.join(output_dir, 'embed.pt')
        torch.save(out, out_path)
        print('File saved to "{}"'.format(out_path))


# DDG-added
@TRAINER_REGISTRY.register()
class Vanilla3(TrainerX_dfed):
    """Vanilla baseline.

    Slightly modified for mixstyle and Decentralized version

    """

    def __init__(self, cfg):
        super().__init__(cfg)
        mix = cfg.TRAINER.VANILLA3.MIX

        for model in self.models:
            if mix == 'random':
                model.apply(random_mixstyle)
                print('MixStyle: random mixing')

            elif mix == 'crossdomain':
                model.apply(crossdomain_mixstyle)
                print('MixStyle: cross-domain mixing')

            else:
                raise NotImplementedError

    # Decentralized (Inspired by the StableFDG paper)
    def forward_backward(self, batches):
        # DDG-edited
        losses, labels = {}, {}
        names = self.get_model_names()

        # DDG-added (Style calculation step)
        for i in range(self.num_clients):
            name = names[i]
            batch = batches[i]
            model = self.models[i]
            input, labels[name] = self.parse_batch_train(batch)

            with torch.no_grad():
                model.eval()
                model(input, style_step=True)
                model.train()
        self.styles_backward_and_update_dfed()

        for i in range(self.num_clients):
            name = names[i]
            batch = batches[i]
            model = self.models[i]
            input, labels[name] = self.parse_batch_train(batch)

            # DDG-edited
            if model.style_sharing and model.style_Sigma_sharing and model.alternate_layers:
                self.outputs[name] = model(input, mu_avg=self._mus[name], sig_avg=self._sigs[name], Sigma_mu_avg=self._Sigma_mus[name], Sigma_sig_avg=self._Sigma_sigs[name], alternate_layer=True)
            elif model.style_sharing and model.style_Sigma_sharing:
                # self.outputs[name] = model(input, mu_avg=self._mus[name], sig_avg=self._sigs[name], Sigma_mu_avg=self._Sigma_mus[name], Sigma_sig_avg=self._Sigma_sigs[name])
                # Style bank version
                # self.outputs[name] = model(input, mu_avg=self._style_bank[name]["mus"], sig_avg=self._style_bank[name]["sigs"], Sigma_mu_avg=self._style_bank[name]["Sigma_mus"], Sigma_sig_avg=self._style_bank[name]["Sigma_sigs"])
                mus = {neighbor_name: value for neighbor_name, value in self._style_bank[name]["mus"].items() if neighbor_name != name}
                sigs = {neighbor_name: value for neighbor_name, value in self._style_bank[name]["sigs"].items() if neighbor_name != name}
                Sigma_mus = {neighbor_name: value for neighbor_name, value in self._style_bank[name]["Sigma_mus"].items() if neighbor_name != name}
                Sigma_sigs = {neighbor_name: value for neighbor_name, value in self._style_bank[name]["Sigma_sigs"].items() if neighbor_name != name}
                self.outputs[name] = model(input, mu_avg=mus, sig_avg=sigs, Sigma_mu_avg=Sigma_mus, Sigma_sig_avg=Sigma_sigs)
            elif model.style_sharing and model.alternate_layers:
                self.outputs[name] = model(input, mu_avg=self._mus[name], sig_avg=self._sigs[name], alternate_layer=True)
            elif model.style_sharing:
                self.outputs[name] = model(input, mu_avg=self._mus[name], sig_avg=self._sigs[name])
            else:
                self.outputs[name] = model(input)
            losses[name] = F.cross_entropy(self.outputs[name], labels[name])
        self.model_backward_and_update_dfed(losses)

        loss_summaries = {name : {
            'loss': losses[name].item(),
            'acc': compute_accuracy(self.outputs[name], labels[name])[0].item()
        } for name in names}

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summaries

    def parse_batch_train(self, batch):
        input = batch['img']
        label = batch['label']
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label
    
    @torch.no_grad()
    def vis(self):
        self.set_model_mode('eval')
        output_dir = self.cfg.OUTPUT_DIR
        source_domains = self.cfg.DATASET.SOURCE_DOMAINS
        print('Source domains:', source_domains)

        # DDG-edited
        out_embeds, out_domains, out_labels = {}, {}, {}

        split = self.cfg.TEST.SPLIT
        data_loader = self.val_loader if split == 'val' else self.test_loader

        print('Extracting style features')

        for batch_idx, batch in enumerate(data_loader):
            input = batch['img'].to(self.device)
            label = batch['label']
            domain = batch['domain']
            impath = batch['impath']

            # model should directly output features or style statistics
            raise NotImplementedError

            # DDG-edited
            names = self.get_model_names()
            for name in names:
                output = model(input)
                output = output.cpu().numpy()
                out_embeds[name].append(output)
                out_domains[name].append(domain.numpy())
                out_labels[name].append(label.numpy()) # CLASS LABEL

            print('processed batch-{}'.format(batch_idx + 1))

        out_embeds = {name : np.concatenate(out_embeds[name], axis=0) for name in names}
        out_domains = {name : np.concatenate(out_domains[name], axis=0) for name in names}
        out_labels = {name : np.concatenate(out_labels[name], axis=0) for name in names}
        print('shape of feature matrix:', out_embeds[0].shape)
        out = {
            'embed': out_embeds,
            'domain': out_domains,
            'dnames': source_domains,
            'label': out_labels
        }
        out_path = osp.join(output_dir, 'embed.pt')
        torch.save(out, out_path)
        print('File saved to "{}"'.format(out_path))
