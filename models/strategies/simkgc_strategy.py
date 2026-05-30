"""SimKGC contrastive training strategy"""

import json
import os

import torch
import torch.nn as nn

from base.evaluator import Evaluator
from base.trainer import Trainer
from data.dict_hub import build_tokenizer, get_entity_dict
from metrics.ranking import topk_accuracy as accuracy
from models.builder import import_module_from_path, load_attr_from_path
from utils.device import get_model_obj, move_to_cuda
from utils.logger import AverageMeter, ProgressMeter, logger


class SimKGCStrategy(Trainer, Evaluator):
    """Training strategy for SimKGC model, implementing contrastive learning with in-batch negatives, pre-batch negatives, and self-negatives, along with evaluation on validation set and link prediction metrics."""

    def __init__(self, args, ngpus_per_node):
        Evaluator.__init__(self)
        self.args = args
        self.ngpus_per_node = ngpus_per_node
        build_tokenizer(args)

        # Load encoder/build_model factory from configured path (fallback to bert encoder)
        encoder_path = getattr(args, 'model_encoder_path', '') or 'models/encoders/bert_encoder.py'
        try:
            build_model = load_attr_from_path(encoder_path, 'build_model')
        except Exception:
            # try importing module and looking for build_model
            mod = import_module_from_path(encoder_path)
            build_model = getattr(mod, 'build_model')

        logger.info('=> creating model')
        model = build_model(args)
        logger.info(model)

        criterion = nn.CrossEntropyLoss()
        if torch.cuda.is_available():
            criterion = criterion.cuda()

        super().__init__(args, ngpus_per_node, model=model, criterion=criterion)

        # load loss helpers
        loss_path = getattr(args, 'model_loss_path', '') or 'models/losses/infonce_loss.py'
        try:
            self.ModelOutput = load_attr_from_path(loss_path, 'ModelOutput')
            self.compute_infonce_logits = load_attr_from_path(loss_path, 'compute_infonce_logits')
        except Exception:
            loss_mod = import_module_from_path(loss_path)
            self.ModelOutput = getattr(loss_mod, 'ModelOutput')
            self.compute_infonce_logits = getattr(loss_mod, 'compute_infonce_logits')

        # load sampler helpers such as construct_mask
        sampler_path = getattr(args, 'model_sampler_path', '') or 'models/samplers/masking_sampler.py'
        try:
            self.construct_mask = load_attr_from_path(sampler_path, 'construct_mask')
        except Exception:
            sampler_mod = import_module_from_path(sampler_path)
            self.construct_mask = getattr(sampler_mod, 'construct_mask')

    def _compute_logits(self, output_dict: dict, batch_dict: dict) -> dict:
        """Compute logits and labels for InfoNCE loss based on model outputs and batch information, applying necessary masking and adjustments for in-batch negatives, pre-batch negatives, and self-negatives."""

        model = get_model_obj(self.model)
        hr_vector, tail_vector = output_dict['hr_vector'], output_dict['tail_vector']
        batch_size = hr_vector.size(0)
        labels = torch.arange(batch_size, device=hr_vector.device)

        logits = self.compute_infonce_logits(
            query_vec=hr_vector,
            candidate_vec=tail_vector,
            temp=model.log_inv_t,
            margin=model.add_margin if model.training else 0.0,
        )

        triplet_mask = batch_dict.get('triplet_mask', None)
        if triplet_mask is not None:
            logits.masked_fill_(~triplet_mask.to(hr_vector.device), -1e4)

        if model.pre_batch > 0 and model.training:
            pre_batch_logits = self._compute_pre_batch_logits(model, hr_vector, tail_vector, batch_dict)
            logits = torch.cat([logits, pre_batch_logits], dim=-1)

        if self.args.use_self_negative and model.training:
            head_vector = output_dict['head_vector']
            self_neg_logits = torch.sum(hr_vector * head_vector, dim=1) * model.log_inv_t.exp()
            self_negative_mask = batch_dict.get('self_negative_mask', None)
            if self_negative_mask is None:
                self_negative_mask = torch.ones(batch_size, dtype=torch.bool, device=hr_vector.device)
            else:
                self_negative_mask = self_negative_mask.to(hr_vector.device).bool()
            self_neg_logits.masked_fill_(~self_negative_mask, -1e4)
            logits = torch.cat([logits, self_neg_logits.unsqueeze(1)], dim=-1)

        return {
            'logits': logits,
            'labels': labels,
            'inv_t': model.log_inv_t.detach().exp(),
            'hr_vector': hr_vector.detach(),
            'tail_vector': tail_vector.detach(),
        }

    def _compute_pre_batch_logits(self, model, hr_vector: torch.Tensor, tail_vector: torch.Tensor, batch_dict: dict) -> torch.Tensor:
        """Compute logits against pre-batch negatives stored in the model, applying necessary masking based on previous batch examples."""

        assert tail_vector.size(0) == model.batch_size
        batch_exs = batch_dict['batch_data']
        pre_batch_logits = self.compute_infonce_logits(hr_vector, model.pre_batch_vectors.clone(), model.log_inv_t)
        pre_batch_logits *= model.args.pre_batch_weight
        if model.pre_batch_exs[-1] is not None:
            pre_triplet_mask = self.construct_mask(batch_exs, model.pre_batch_exs).to(hr_vector.device)
            pre_batch_logits.masked_fill_(~pre_triplet_mask, -1e4)

        model.pre_batch_vectors[model.offset:(model.offset + model.batch_size)] = tail_vector.data.clone()
        model.pre_batch_exs[model.offset:(model.offset + model.batch_size)] = batch_exs
        model.offset = (model.offset + model.batch_size) % len(model.pre_batch_exs)

        return pre_batch_logits

    def train_epoch(self, epoch) -> None:
        """Train the model for one epoch, iterating over the training data, computing losses and accuracies, and performing optimization steps, while periodically evaluating on the validation set."""

        losses = AverageMeter('Loss', ':.4')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top3 = AverageMeter('Acc@3', ':6.2f')
        inv_t = AverageMeter('InvT', ':6.2f')
        progress = ProgressMeter(
            len(self.train_loader),
            [losses, inv_t, top1, top3],
            prefix='Epoch: [{}]'.format(epoch),
        )

        for i, batch_dict in enumerate(self.train_loader):
            self.model.train()

            if torch.cuda.is_available():
                batch_dict = move_to_cuda(batch_dict)
            batch_size = len(batch_dict['batch_data'])

            if self.args.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(**batch_dict)
            else:
                outputs = self.model(**batch_dict)

            outputs = self._compute_logits(output_dict=outputs, batch_dict=batch_dict)
            outputs = self.ModelOutput(**outputs)
            logits, labels = outputs.logits, outputs.labels
            assert logits.size(0) == batch_size

            loss = self.criterion(logits, labels)
            loss += self.criterion(logits[:, :batch_size].t(), labels)

            acc1, acc3 = accuracy(logits, labels, topk=(1, 3))
            top1.update(acc1.item(), batch_size)
            top3.update(acc3.item(), batch_size)
            inv_t.update(outputs.inv_t, 1)
            losses.update(loss.item(), batch_size)

            self.optimizer.zero_grad()
            if self.args.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
                self.optimizer.step()
            self.scheduler.step()

            if i % self.args.print_freq == 0:
                progress.display(i)
            if (i + 1) % self.args.eval_every_n_step == 0:
                self._run_eval(epoch=epoch, step=i + 1)

        logger.info('Learning rate: {}'.format(self.scheduler.get_last_lr()[0]))
        log_str = f"[EPOCH {epoch}] Loss: {losses.avg:.4f} | Acc@1: {top1.avg:.2f} | Acc@3: {top3.avg:.2f}"
        print(log_str)
        logger.info(log_str)

    @torch.no_grad()
    def eval_epoch(self, epoch) -> dict:
        """Evaluate the model on the validation set for one epoch, computing metrics such as loss, accuracy, and MRR for link prediction, and return a dictionary of these metrics."""

        metric_dict = {}
        if self.valid_loader:
            losses = AverageMeter('Loss', ':.4')
            top1 = AverageMeter('Acc@1', ':6.2f')
            top3 = AverageMeter('Acc@3', ':6.2f')

            for _, batch_dict in enumerate(self.valid_loader):
                self.model.eval()
                if torch.cuda.is_available():
                    batch_dict = move_to_cuda(batch_dict)
                batch_size = len(batch_dict['batch_data'])

                outputs = self.model(**batch_dict)
                outputs = self._compute_logits(output_dict=outputs, batch_dict=batch_dict)
                outputs = self.ModelOutput(**outputs)
                logits, labels = outputs.logits, outputs.labels
                loss = self.criterion(logits, labels)
                losses.update(loss.item(), batch_size)

                acc1, acc3 = accuracy(logits, labels, topk=(1, 3))
                top1.update(acc1.item(), batch_size)
                top3.update(acc3.item(), batch_size)

            metric_dict.update({
                'Acc@1': round(top1.avg, 3),
                'Acc@3': round(top3.avg, 3),
                'loss': round(losses.avg, 3),
            })

        valid_eval_path = None
        if self.args.valid_path:
            if self.args.valid_path.endswith('_w_label.txt'):
                cand_txt = self.args.valid_path.replace('valid_w_label.txt', 'valid.txt')
                cand_json = self.args.valid_path.replace('valid_w_label.txt', 'valid.txt.json')
                if os.path.exists(cand_json):
                    valid_eval_path = cand_json
                elif os.path.exists(cand_txt):
                    valid_eval_path = cand_txt
            elif self.args.valid_path.endswith('.txt.json') or self.args.valid_path.endswith('.txt'):
                valid_eval_path = self.args.valid_path

        if valid_eval_path and os.path.exists(valid_eval_path):
            valid_entity_dict = get_entity_dict()
            valid_output_path = os.path.join(self.args.model_dir, 'valid_link_prediction.log')
            forward_metrics = self.evaluate_link_prediction_inplace(
                self.model, valid_eval_path, valid_entity_dict, valid_output_path, eval_forward=True)
            backward_metrics = self.evaluate_link_prediction_inplace(
                self.model, valid_eval_path, valid_entity_dict, valid_output_path, eval_forward=False)
            if forward_metrics and backward_metrics:
                metric_dict['mrr'] = round((forward_metrics.get('mrr', 0) + backward_metrics.get('mrr', 0)) / 2, 4)
                logger.info(f"[EPOCH {epoch}] Validation link-pred MRR(avg): {metric_dict['mrr']}")

        if metric_dict:
            logger.info('Epoch {}, valid metric: {}'.format(epoch, json.dumps(metric_dict)))
        return metric_dict