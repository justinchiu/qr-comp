from __future__ import annotations

import torch


def _require_square_batch(a: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if a.ndim == 2:
        if a.shape[0] != a.shape[1]:
            raise ValueError("input must be square")
        return a.unsqueeze(0), True
    if a.ndim == 3:
        if a.shape[-2] != a.shape[-1]:
            raise ValueError("input matrices must be square")
        return a, False
    raise ValueError("input must have shape (n, n) or (batch, n, n)")


def _householder_vector(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return compact Householder pieces for each row vector in x.

    For each input vector x, this returns beta, tau, and the stored tail of v
    where H = I - tau * v * v.T and v[0] is implicit 1.
    """
    alpha = x[:, 0]
    tail = x[:, 1:]
    norm = torch.linalg.vector_norm(x, dim=1)
    sign = torch.where(alpha >= 0, torch.ones_like(alpha), -torch.ones_like(alpha))
    beta = -sign * norm

    zero = norm == 0
    beta = torch.where(zero, alpha, beta)

    denom = alpha - beta
    denom_safe = torch.where(denom != 0, denom, torch.ones_like(denom))
    v_tail = tail / denom_safe[:, None]
    tau = torch.where(beta != 0, (beta - alpha) / beta, torch.zeros_like(beta))
    v_tail = torch.where(tau[:, None] != 0, v_tail, torch.zeros_like(v_tail))
    return beta, tau, v_tail


def _apply_reflector_left(
    c: torch.Tensor,
    v_tail: torch.Tensor,
    tau: torch.Tensor,
) -> None:
    if c.shape[-1] == 0:
        return
    batch, rows, _ = c.shape
    v = torch.zeros((batch, rows), dtype=c.dtype, device=c.device)
    v[:, 0] = 1
    v[:, 1:] = v_tail
    dot = torch.bmm(v[:, None, :], c).squeeze(1)
    c -= v[:, :, None] * (tau[:, None] * dot)[:, None, :]


def compact_householder_qr(a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Unblocked compact Householder QR.

    This is the readable reference implementation. It returns the same contract
    as torch.geqrf: lower(H) stores Householder tails, triu(H) stores R, and tau
    stores reflector coefficients.
    """
    a_batch, squeeze = _require_square_batch(a)
    h = a_batch.clone()
    batch, n, _ = h.shape
    tau = torch.zeros((batch, n), dtype=h.dtype, device=h.device)

    for k in range(n):
        beta, tau_k, v_tail = _householder_vector(h[:, k:, k])
        h[:, k, k] = beta
        h[:, k + 1 :, k] = v_tail
        tau[:, k] = tau_k
        _apply_reflector_left(h[:, k:, k + 1 :], v_tail, tau_k)

    if squeeze:
        return h[0], tau[0]
    return h, tau


def _form_panel_v(h: torch.Tensor, panel_start: int, panel_width: int) -> torch.Tensor:
    batch, n, _ = h.shape
    rows = n - panel_start
    v = torch.zeros((batch, rows, panel_width), dtype=h.dtype, device=h.device)
    for j in range(panel_width):
        v[:, j, j] = 1
        v[:, j + 1 :, j] = h[:, panel_start + j + 1 :, panel_start + j]
    return v


def _form_wy_t(v: torch.Tensor, tau_panel: torch.Tensor) -> torch.Tensor:
    batch, _, width = v.shape
    t = torch.zeros((batch, width, width), dtype=v.dtype, device=v.device)
    for i in range(width):
        tau_i = tau_panel[:, i]
        if i:
            prev_v = v[:, :, :i]
            v_i = v[:, :, i : i + 1]
            col = -tau_i[:, None] * torch.bmm(prev_v.transpose(1, 2), v_i).squeeze(-1)
            col = torch.bmm(t[:, :i, :i], col[:, :, None]).squeeze(-1)
            t[:, :i, i] = col
        t[:, i, i] = tau_i
    return t


def _apply_block_reflector_left(c: torch.Tensor, v: torch.Tensor, t: torch.Tensor) -> None:
    if c.shape[-1] == 0:
        return
    work = torch.bmm(v.transpose(1, 2), c)
    work = torch.bmm(t.transpose(1, 2), work)
    c -= torch.bmm(v, work)


def blocked_compact_householder_qr(
    a: torch.Tensor,
    block_size: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Blocked compact Householder QR with a compact-WY trailing update.

    The panel factorization is still scalar Householder. At the end of each
    panel, the accumulated reflectors are applied to the trailing matrix as
    I - V T.T V.T, which mirrors the shape of a GPU blocked implementation.
    """
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    a_batch, squeeze = _require_square_batch(a)
    h = a_batch.clone()
    batch, n, _ = h.shape
    tau = torch.zeros((batch, n), dtype=h.dtype, device=h.device)

    for panel_start in range(0, n, block_size):
        panel_width = min(block_size, n - panel_start)
        panel_end = panel_start + panel_width

        for j in range(panel_width):
            k = panel_start + j
            beta, tau_k, v_tail = _householder_vector(h[:, k:, k])
            h[:, k, k] = beta
            h[:, k + 1 :, k] = v_tail
            tau[:, k] = tau_k
            _apply_reflector_left(h[:, k:, k + 1 : panel_end], v_tail, tau_k)

        v = _form_panel_v(h, panel_start, panel_width)
        t = _form_wy_t(v, tau[:, panel_start:panel_end])
        _apply_block_reflector_left(h[:, panel_start:, panel_end:], v, t)

    if squeeze:
        return h[0], tau[0]
    return h, tau
