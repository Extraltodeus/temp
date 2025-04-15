import torch

def slerp(v2, v1, t=0.5):
    v1_norm = v1 / torch.linalg.norm(v1)
    v2_norm = v2 / torch.linalg.norm(v2)
    dot_product = torch.dot(v1_norm, v2_norm)
    dot_product = torch.clamp(dot_product, min=-1.0, max=1.0)
    theta = torch.arccos(dot_product)
    sin_theta = torch.sin(theta)
    if sin_theta == 0:
        return (1 - t) * v1 + t * v2
    factor1 = torch.sin((1 - t) * theta) / sin_theta
    factor2 = torch.sin(t * theta) / sin_theta
    result = factor1 * v1 + factor2 * v2
    return result

def manage_duplicates(batch):
    unique_elements, inverse_indices, counts = torch.unique(batch, return_inverse=True, return_counts=True, dim=0)
    occurrences = counts[inverse_indices]
    sorted_inverse, sorted_indices = torch.sort(inverse_indices)
    diff = sorted_inverse[1:] != sorted_inverse[:-1]
    group_boundaries = torch.cat([
        torch.tensor([0], device=diff.device),
        (torch.where(diff)[0] + 1)
    ])
    first_occurrence_indices = sorted_indices[group_boundaries]
    mask = torch.zeros_like(inverse_indices, dtype=torch.bool)
    mask[first_occurrence_indices] = True
    return occurrences, mask

def check_return(t, w=None):
    if t.ndim == 1:
        return t, True
    if t.shape[0] == 1:
        return t[0], True
    if t.shape[0] == 2:
        if w is not None:
            return slerp(t[0], t[1], w[0] / (w[0] + w[1])), True
        return slerp(t[0], t[1]), True
    return t, False

def spherical_batch_interpolation(t, w=None, target_magnitude=None, weighted_mag=False, *args, **kwargs):
    """
    Takes a batch of vectors 't' of shape [batch, ndim] and optional weights 'w'.
    The weights can be a simple array, a batch of scalars or even be the same shape as the input batch.
    Allowing to weight each individual dimension if needed.
    returns a single vector of shape [ndim]
    """
    t, should_return = check_return(t, w)
    if should_return:
        return t

    w_none = w == None
    if not w_none:
        if not isinstance(w, torch.Tensor):
            w = torch.tensor(w)
        while w.ndim < t.ndim:
            w = w.unsqueeze(-1)
        if w.device != t.device:
            w = w.to(t.device)
        w = w / w.sum(dim=0)
    else:
        w = torch.ones(t.shape[0], 1, device=t.device) / t.shape[0]

    occ, singles = manage_duplicates(t)
    if torch.any(occ > 1):
        w_none = False
        occ = occ.unsqueeze(-1)
        w = w * occ
        w = w[singles]
        w = w / w.sum(dim=0)
        t = t[singles]

    batch_size = t.shape[0]
    norms = torch.linalg.norm(t, dim=1, keepdim=True)
    v = t / norms
    dots = torch.mm(v, v.T).clamp(min=-1.0, max=1.0)

    mask = ~torch.eye(batch_size, dtype=torch.bool, device=t.device)
    dots = dots[mask].reshape(batch_size, batch_size - 1)

    omegas = dots.acos()
    sin_omega = omegas.sin()

    # repeats each vector and respective weights / (batch_size - 1) so each uses it's own dot products versus every other vector and have it's own weight divided by as many versus.
    # then recombine them using their sum on the repeated axis for each duplicated version and on the last axis to complete the weighted average.
    res = t.unsqueeze(1).repeat(1, batch_size - 1, 1) * torch.sin(w.div(batch_size - 1).unsqueeze(1).repeat(1, batch_size - 1, 1) * omegas.unsqueeze(-1)) / sin_omega.unsqueeze(-1)
    res = res.sum(dim=[0, 1])
    return res

# this one is for latent spaces of dim [batch, channel, y, x]
# t is the batch of latents, tn is the batch of latents divided by it's norm so torch.linalg.matrix_norm(t, keepdim=True)
# and w are the weights, shaped like the batch of latents
@torch.no_grad()
def matrix_batch_slerp(t, tn, w):
    dots = torch.mul(tn.unsqueeze(0), tn.unsqueeze(1)).sum(dim=[-1,-2], keepdim=True).clamp(min=-1.0, max=1.0)
    mask = ~torch.eye(t.shape[0], dtype=torch.bool, device=t.device)
    A, B, C, D, E = dots.shape
    dots = dots[mask].reshape(A, B - 1, C, D, E)
    omegas = dots.acos()
    sin_omega = omegas.sin()
    res = t.unsqueeze(1).repeat(1, B - 1, 1, 1, 1) * torch.sin(w.div(B - 1).unsqueeze(1).repeat(1, B - 1, 1, 1, 1) * omegas) / sin_omega
    res = res.sum(dim=[0, 1]).unsqueeze(0)
    return res
