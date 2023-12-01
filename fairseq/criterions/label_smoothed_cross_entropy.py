# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion

import math
import torch
import torch.nn.functional as F

def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss

def cost(x, y):
    len1 = x.size(-2)
    len2 = y.size(-2)
    dim = x.size(-1)
    bsz = x.size(0)
    tx = x.unsqueeze(dim=-2).expand(bsz, len1, len2, dim)
    ty = y.unsqueeze(dim=-3).expand(bsz, len1, len2, dim)
    res = torch.linalg.norm(tx - ty, dim=-1)
    return res

def compute_ot_loss(x, y):
    y = y.view(y.size(-1), -1, y.size(0))
    x = x.view(x.size(-1), -1, x.size(0))
    y = F.normalize(y, p=2, dim=0, eps=1e-5)
    x = F.normalize(x, p=2, dim=0, eps=1e-5)
    y = y.transpose(0, 1)
    x = x.transpose(0, 1)
    C1 = cost(x, y)
    weight1 = torch.linalg.norm(x, dim=-1) / torch.linalg.norm(x, dim=-1).sum(dim=-1, keepdim=True)
    res1 = (C1.min(dim=-1)[0] * weight1.detach().clone()).sum()
    C2 = cost(x, y)
    weight2 = torch.linalg.norm(y, dim=-1) / torch.linalg.norm(y, dim=-1).sum(dim=-1,keepdim=True)
    res2 = (C2.min(dim=-1)[0] * weight2.detach().clone()).sum()
    loss = 0.5 * (res1 + res2)
    return loss

def kl_div_loss(lprobs_p, lprobs_q, target, ignore_index=None, reduce=True):
    '''
    Kullback–Leibler divergence between probability distributions
    '''
    loss = F.kl_div(lprobs_p, lprobs_q.exp(), reduction='none')
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        loss.masked_fill_(pad_mask, 0.0)
    else:
        loss = loss.squeeze(-1)
    if reduce:
        loss = loss.sum()
    return loss

@register_criterion("label_smoothed_cross_entropy")
class LabelSmoothedCrossEntropyCriterion(FairseqCriterion):
    def __init__(
            self,
            task,
            sentence_avg,
            label_smoothing,
            ignore_prefix_size=0,
            report_accuracy=False,
    ):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.ignore_prefix_size = ignore_prefix_size
        self.report_accuracy = report_accuracy

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        parser.add_argument('--report-accuracy', action='store_true',
                            help='report accuracy metric')
        parser.add_argument('--ignore-prefix-size', default=0, type=int,
                            help='Ignore first N tokens')
        # fmt: on
        parser.add_argument('--kl-weight', action='store_true',
                            help='weight of the kl loss')
        parser.add_argument('--ot-weight', default=0, type=int,
                            help='weight of the ot loss')

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        encoder_out_aut = model.encoder(sample['net_input']['src_tokens'], sample['net_input']['src_lengths'],
                                        sample['net_input']['authe_imgs_list'], sample['net_input']['authe_masks_list'],
                                        sample['net_input']['synth_imgs_list'], sample['net_input']['synth_masks_list'],
                                        ffn_net=model.ffn_net, image_encoder=model.image_encoder)
        encoder_out_syn = model.encoder(sample['net_input']['src_tokens'], sample['net_input']['src_lengths'],
                                        sample['net_input']['synth_imgs_list'], sample['net_input']['synth_masks_list'],
                                        sample['net_input']['authe_imgs_list'], sample['net_input']['authe_masks_list'],
                                        ffn_net=model.ffn_net, image_encoder=model.image_encoder)
        net_output = model(**sample["net_input"], ffn_net=model.ffn_net, image_encoder=model.image_encoder)
        loss, nll_loss, loss_dict = self.compute_loss(model, net_output, sample, encoder_out_aut, encoder_out_syn,
                                                      reduce=reduce)
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "syn_loss": loss_dict['syn'].data if 'syn' in loss_dict else 0,
            "aut_loss": loss_dict['aut'].data if 'aut' in loss_dict else 0,
            "kl_loss": loss_dict['kl'].data if 'kl' in loss_dict else 0,
            'ot_loss': loss_dict['ot_loss'].data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        return loss.float(), sample_size, logging_output

    def get_lprobs_and_target(self, model, net_output, sample):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        target = model.get_targets(sample, net_output)
        if self.ignore_prefix_size > 0:
            if getattr(lprobs, "batch_first", False):
                lprobs = lprobs[:, self.ignore_prefix_size:, :].contiguous()
                target = target[:, self.ignore_prefix_size:].contiguous()
            else:
                lprobs = lprobs[self.ignore_prefix_size:, :, :].contiguous()
                target = target[self.ignore_prefix_size:, :].contiguous()
        return lprobs.view(-1, lprobs.size(-1)), target.view(-1)

    def compute_loss(self, model, net_output, sample, encoder_out_aut, encoder_out_syn, reduce=True):
        loss_dict = {}
        syn = 'syn' in net_output and net_output['syn'] is not None
        aut = 'aut' in net_output and net_output['aut'] is not None
        assert syn or aut, "Net output is empty"
        if syn:
            lprobs_syn, target = self.get_lprobs_and_target(model, net_output['syn'], sample)
            loss_dict['syn'], loss_dict['syn_nll'] = label_smoothed_nll_loss(
                lprobs_syn,
                target,
                self.eps,
                ignore_index=self.padding_idx,
                reduce=reduce,
            )
            loss = loss_dict['syn']
            nll_loss = loss_dict['syn_nll']
        if aut:
            lprobs_aut, target = self.get_lprobs_and_target(model, net_output['aut'], sample)
            loss_dict['aut'], loss_dict['aut_nll'] = label_smoothed_nll_loss(
                lprobs_aut,
                target,
                self.eps,
                ignore_index=self.padding_idx,
                reduce=reduce,
            )
            loss = (loss + loss_dict['aut']) / 2 if loss is not None else loss_dict['aut']
            nll_loss = (nll_loss + loss_dict['aut_nll']) / 2 if nll_loss is not None else loss_dict['aut_nll']
        self.kl_weight = 0.5
        self.ot_weight = 0.1
        if syn and aut:
            if self.kl_weight > 1e-6:
                kl_loss = kl_div_loss(lprobs_aut, lprobs_syn, target)
                loss_dict['kl'] = kl_loss
                loss += self.kl_weight * kl_loss
        loss_dict['ot_loss'] = compute_ot_loss(encoder_out_aut.img_feature,encoder_out_syn.img_feature)
        if self.ot_weight > 1e-6:
            loss += self.ot_weight * loss_dict['ot_loss']
        return loss, nll_loss, loss_dict

    def compute_accuracy(self, model, net_output, sample):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        mask = target.ne(self.padding_idx)
        n_correct = torch.sum(
            lprobs.argmax(1).masked_select(mask).eq(target.masked_select(mask))
        )
        total = torch.sum(mask)
        return n_correct, total

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        syn_loss_sum = sum(log.get("syn_loss", 0) for log in logging_outputs)
        aut_loss_sum = sum(log.get("aut_loss", 0) for log in logging_outputs)
        kl_loss_sum = sum(log.get("kl_loss", 0) for log in logging_outputs)
        ot_loss_sum = sum(log.get("ot_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_scalar(
            "syn_loss", syn_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_scalar(
            "aut_loss", aut_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_scalar(
            "kl_loss", kl_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_scalar(
            "ot_loss", ot_loss_sum / nsentences / math.log(2), nsentences, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        )

        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True