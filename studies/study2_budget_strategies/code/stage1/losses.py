"""Stage-1 auxiliary objectives (protocol v1.0, Decisions 1-3), written for per-batch tensors.

Every function takes the batch logits / representations plus integer metadata and
returns a scalar tensor (0 when the batch contains nothing the term applies to).

Family codes: 0 Clean (incl. clean repeats), 1 RT, 2 RN, 3 AD, 4 TI (target intensification).
``group`` holds, per item, the index of its source group inside the batch, or -1.
``anchor`` marks the clean source row of a group.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

CLEAN, RT, RN, AD, TI = 0, 1, 2, 3, 4
FAMILY_CODE = {"Clean": CLEAN, "CleanRepeat": CLEAN, "RT": RT, "RN": RN, "AD": AD, "TI": TI}


def _groups(group: torch.Tensor, anchor: torch.Tensor):
    """Yield (anchor_index, {family: [indices]}) for each group that has an anchor."""
    out = {}
    for i, (g, a) in enumerate(zip(group.tolist(), anchor.tolist())):
        if g < 0:
            continue
        out.setdefault(g, {"anchor": None, "members": []})
        if a:
            out[g]["anchor"] = i
        else:
            out[g]["members"].append(i)
    for g in out.values():
        if g["anchor"] is not None and g["members"]:
            yield g["anchor"], g["members"]


def js_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Symmetric Jensen-Shannon divergence between probability vectors (last dim)."""
    m = 0.5 * (p + q)
    kl_pm = (p * (torch.log(p + eps) - torch.log(m + eps))).sum(-1)
    kl_qm = (q * (torch.log(q + eps) - torch.log(m + eps))).sum(-1)
    return 0.5 * kl_pm + 0.5 * kl_qm


def invariance_loss(logits, family, group, anchor, families=(RN, AD)) -> torch.Tensor:
    """Protocol 3.5 L_I: per source, mean of JS(p_x, p_member) over its label-preserving members; mean over sources."""
    probs = torch.softmax(logits.float(), dim=-1)
    terms = []
    for a, members in _groups(group, anchor):
        js = [js_divergence(probs[a], probs[m]) for m in members if int(family[m]) in families]
        if js:
            terms.append(torch.stack(js).mean())
    return torch.stack(terms).mean() if terms else logits.new_zeros((), dtype=torch.float32)


def sensitivity_loss(logits, family, labels, orig_labels, margin: float) -> torch.Tensor:
    """Protocol 3.5 L_S: hinge on z_{y'}(x_RT) - z_y(x_RT) for every RT record; mean over RT records."""
    idx = (family == RT).nonzero(as_tuple=True)[0]
    if idx.numel() == 0:
        return logits.new_zeros((), dtype=torch.float32)
    z = logits.float()[idx]
    gap = z.gather(1, labels[idx, None]).squeeze(1) - z.gather(1, orig_labels[idx, None]).squeeze(1)
    return F.relu(margin - gap).mean()


def bat_triplet_loss(reps, family, group, anchor, margin: float) -> torch.Tensor:
    """Decision 1: [d(h_x, h_TI) - d(h_x, h_RT) + m]_+ with cosine distance on L2-normalised [CLS]."""
    h = F.normalize(reps.float(), dim=-1)
    terms = []
    for a, members in _groups(group, anchor):
        pos = [m for m in members if int(family[m]) == TI]
        neg = [m for m in members if int(family[m]) == RT]
        if pos and neg:
            d_pos = 1 - (h[a] * h[pos[0]]).sum()
            d_neg = 1 - (h[a] * h[neg[0]]).sum()
            terms.append(F.relu(d_pos - d_neg + margin))
    return torch.stack(terms).mean() if terms else reps.new_zeros((), dtype=torch.float32)


def fei_opinion_loss(reps, family, group, anchor, mu: float, infonce: bool = False) -> torch.Tensor:
    """Decision 2: intra (negative RT) + inter (negative AD) terms, positive TI, anchor = clean source.

    Protocol v1.0 form: -log( exp(Sim(TI,x)/mu) / sum_k exp(Sim(neg_k,x)/mu) ), summed over the two
    terms, mean over sources. With ``infonce=True`` (amendment option, see IMPLEMENTATION_NOTES N20)
    the positive is also in the denominator, the usual InfoNCE form, which saturates.
    """
    h = F.normalize(reps.float(), dim=-1)
    terms = []
    for a, members in _groups(group, anchor):
        pos = [m for m in members if int(family[m]) == TI]
        if not pos:
            continue
        s_pos = (h[a] * h[pos[0]]).sum() / mu
        total = None
        for neg_family in (RT, AD):
            negs = [m for m in members if int(family[m]) == neg_family]
            if negs:
                s_neg = torch.stack([(h[a] * h[m]).sum() / mu for m in negs])
                if infonce:
                    s_neg = torch.cat([s_pos.reshape(1), s_neg])
                t = torch.logsumexp(s_neg, dim=0) - s_pos
                total = t if total is None else total + t
        if total is not None:
            terms.append(total)
    return torch.stack(terms).mean() if terms else reps.new_zeros((), dtype=torch.float32)


def per_family_ce(logits, labels, family) -> dict[int, tuple[torch.Tensor, int]]:
    """Mean CE per family present in the batch, with its item count (GroupDRO / PCGrad groups)."""
    ce = F.cross_entropy(logits.float(), labels, reduction="none")
    out = {}
    for f in sorted(set(family.tolist())):
        mask = family == f
        out[f] = (ce[mask].mean(), int(mask.sum()))
    return out


class GroupDROWeights:
    """Online GroupDRO (Sagawa et al., 2020): q_g <- q_g * exp(eta * loss_g), renormalised."""

    def __init__(self, n_groups: int, eta: float):
        self.q = torch.full((n_groups,), 1.0 / n_groups)
        self.eta = eta

    def loss(self, group_losses: dict[int, tuple[torch.Tensor, int]]) -> torch.Tensor:
        q = self.q.clone()
        for g, (lg, _) in group_losses.items():
            q[g] = q[g] * torch.exp(self.eta * lg.detach().float().cpu())
        self.q = q / q.sum()
        return sum(self.q[g].to(lg.device) * lg for g, (lg, _) in group_losses.items())


def pcgrad_combine(grads: list[torch.Tensor], generator: torch.Generator) -> torch.Tensor:
    """PCGrad (Yu et al., 2020): project each task gradient off conflicting ones, then sum."""
    projected = []
    for i, g in enumerate(grads):
        gi = g.clone()
        order = torch.randperm(len(grads), generator=generator).tolist()
        for j in order:
            if j == i:
                continue
            dot = torch.dot(gi, grads[j])
            if dot < 0:
                gi = gi - dot / (grads[j].norm() ** 2 + 1e-12) * grads[j]
        projected.append(gi)
    return torch.stack(projected).sum(0)
