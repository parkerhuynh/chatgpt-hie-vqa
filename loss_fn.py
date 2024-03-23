from torch import nn
import torch

def instance_bce_with_logits(logits, labels):
    assert logits.dim() == 2
    criterion = nn.BCEWithLogitsLoss()
    loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
    # loss = criterion(logits, labels)
    loss *= labels.size(1)
    return loss


def compute_score_with_logits_paddingremoved(args,logits, labels):
    if args.dataset in ["simpsons"]:
        logits = torch.argmax(logits, dim=1)
        final_score = (logits == labels).sum().item()
        sample = logits.size(0)
    else:
        logits = torch.max(logits, 1)[1].data # argmax
        one_hots = torch.zeros(*labels.size()).cuda()
        one_hots.scatter_(1, logits.view(-1, 1), 1)
        scores = (one_hots * labels)

        max_labels = torch.max(labels, 1)[1]

        non_padding_idx = (max_labels != (labels.size(1)-1)).nonzero()

        non_padded = torch.index_select(scores.sum(1), 0, non_padding_idx.squeeze())

        final_score = non_padded.sum().item()
        sample = non_padded.size(0)

    return final_score, sample