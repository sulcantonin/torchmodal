"""Regenerate the figures in docs/figures (light + dark SVG of each).

    python docs/make_figures.py            # all figures
    python docs/make_figures.py kripke     # just one

Every number drawn comes from torchmodal itself, so the pictures stay honest
when the library changes. Needs matplotlib.
"""

import math
import os
import sys
from itertools import product

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import (  # noqa: E402
    Circle,
    FancyArrowPatch,
    FancyBboxPatch,
    Rectangle,
)

# Always draw with the checkout, never a previously installed release.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from torchmodal import functional as F  # noqa: E402
from torchmodal import verify  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")

THEMES = {
    "light": dict(
        surface="#fcfcfb",
        ink="#0b0b0b",
        ink2="#52514e",
        muted="#898781",
        grid="#e1e0d9",
        axis="#c3c2b7",
        border="#e1e0d9",
        box="#f3f2ee",
        accent="#e8492f",
        accent_band="#f9c9bd",
        good="#1baf7a",
        bad="#d63b3b",
        unsure="#eda100",
        blue="#2a78d6",
        seq=[
            "#fcfcfb",
            "#fde3dc",
            "#f9c9bd",
            "#f4a08b",
            "#ee7457",
            "#e8492f",
            "#b8321d",
            "#7e2111",
        ],
    ),
    "dark": dict(
        surface="#1a1a19",
        ink="#f0efec",
        ink2="#c3c2b7",
        muted="#898781",
        grid="#2c2c2a",
        axis="#383835",
        border="#383835",
        box="#242423",
        accent="#f0603f",
        accent_band="#6b2a1c",
        good="#199e70",
        bad="#e05252",
        unsure="#c98500",
        blue="#3987e5",
        seq=[
            "#1a1a19",
            "#33201b",
            "#53291d",
            "#7e3420",
            "#b0412a",
            "#e8492f",
            "#f4876c",
            "#fcc5b6",
        ],
    ),
}


def style(t):
    plt.rcParams.update(
        {
            "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 11,
            "svg.fonttype": "none",
            "svg.hashsalt": "torchmodal",  # stable ids: regenerating gives no diff
            "figure.facecolor": t["surface"],
            "axes.facecolor": t["surface"],
            "savefig.facecolor": t["surface"],
            "text.color": t["ink"],
            "axes.labelcolor": t["ink2"],
            "axes.edgecolor": t["axis"],
            "xtick.color": t["muted"],
            "ytick.color": t["muted"],
            "xtick.labelcolor": t["ink2"],
            "ytick.labelcolor": t["ink2"],
            "axes.titlesize": 12.5,
            "axes.titleweight": "bold",
            "axes.titlecolor": t["ink"],
            "axes.titlelocation": "left",
            "axes.titlepad": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "legend.labelcolor": t["ink2"],
        }
    )


def titled(ax, title, sub=None, t=None):
    """Bold title with an optional grey subtitle underneath, never overlapping."""
    lines = 0 if sub is None else sub.count("\n") + 1
    ax.set_title(title, pad=8 + 14 * lines)
    if sub:
        ax.text(
            0,
            1.02,
            sub,
            transform=ax.transAxes,
            va="bottom",
            fontsize=9.5,
            color=t["ink2"],
        )


def save(fig, name, theme):
    fig.savefig(
        os.path.join(OUT, f"{name}-{theme}.svg"),
        bbox_inches="tight",
        pad_inches=0.25,
        metadata={"Date": None},
    )
    plt.close(fig)


def fmt(lo, hi):
    return f"[{float(lo):.2f}, {float(hi):.2f}]"


def arrow(ax, p, q, color, lw=1.4, r=0.17, rad=0.0, alpha=1.0, z=1):
    """Arrow from node centre p to node centre q, stopping at the node rims."""
    (x0, y0), (x1, y1) = p, q
    d = math.hypot(x1 - x0, y1 - y0)
    ux, uy = (x1 - x0) / d, (y1 - y0) / d
    gap = r * 1.12
    ax.add_patch(
        FancyArrowPatch(
            (x0 + gap * ux, y0 + gap * uy),
            (x1 - gap * ux, y1 - gap * uy),
            arrowstyle="-|>,head_length=6,head_width=3.5",
            connectionstyle=f"arc3,rad={rad}",
            color=color,
            lw=lw,
            shrinkA=0,
            shrinkB=0,
            alpha=alpha,
            zorder=z,
            mutation_scale=1,
        )
    )


def self_loop(ax, p, color, r, lw=1.4, alpha=1.0, up=True):
    s = 1 if up else -1
    x, y = p
    ax.add_patch(
        FancyArrowPatch(
            (x - 0.6 * r, y + s * 0.8 * r),
            (x + 0.6 * r, y + s * 0.8 * r),
            arrowstyle="-|>,head_length=5,head_width=3",
            connectionstyle=f"arc3,rad={-1.9 * s}",
            color=color,
            lw=lw,
            alpha=alpha,
            zorder=1,
            mutation_scale=1,
        )
    )


# ---- shared toy frame: five worlds ---------------------------------------------------

# A small Kripke frame used in sections 1 and 2.
KRIPKE_EDGES = [(0, 1), (0, 2), (1, 3), (2, 4), (3, 2), (3, 3), (4, 0), (4, 1)]
KRIPKE_P = [1.0, 0.0, 1.0, 1.0, 0.0]


def kripke_frame():
    A = torch.zeros(5, 5)
    for a, b in KRIPKE_EDGES:
        A[a, b] = 1.0
    p = torch.tensor(KRIPKE_P).unsqueeze(-1).expand(-1, 2)  # (|W|, 2): [L, U] per world
    return A, p


def pentagon(r=1.0):
    return {
        i: (r * math.sin(2 * math.pi * i / 5), r * math.cos(2 * math.pi * i / 5))
        for i in range(5)
    }


# ---- 0. the big picture --------------------------------------------------------------


def fig_pipeline(t):
    fig, ax = plt.subplots(figsize=(12, 3.1))
    steps = [
        (
            "1 · Worlds",
            "agents, time steps,\ncontexts, cells…\nand a relation A\nbetween them",
        ),
        ("2 · Bounds", "every proposition\nholds in every world\nwithin [L, U]"),
        (
            "3 · □ and ♢",
            "necessity and\npossibility neurons:\nsound intervals\nacross worlds",
        ),
        (
            "4 · Learn",
            "contradiction loss\nL > U trains the\n"
            "propositions and/or\nthe relation A_θ",
        ),
        ("5 · Certify", "round A, evaluate\nexactly: PROVEN /\nREFUTED / UNDECIDED"),
    ]
    W, gap = 1.9, 0.45
    for i, (head, body) in enumerate(steps):
        x = i * (W + gap)
        hl = i == 2
        ax.add_patch(
            FancyBboxPatch(
                (x, 0),
                W,
                2.2,
                boxstyle="round,pad=0,rounding_size=0.12",
                fc=t["accent"] if hl else t["box"],
                ec=t["accent"] if hl else t["border"],
                lw=1.2,
            )
        )
        ink = "#ffffff" if hl else t["ink"]
        ax.text(
            x + 0.15, 1.95, head, fontsize=11.5, fontweight="bold", color=ink, va="top"
        )
        ax.text(
            x + 0.15,
            1.45,
            body,
            fontsize=9.5,
            color="#ffffff" if hl else t["ink2"],
            va="top",
            linespacing=1.45,
        )
        if i < len(steps) - 1:
            ax.annotate(
                "",
                xy=(x + W + gap - 0.04, 1.1),
                xytext=(x + W + 0.04, 1.1),
                arrowprops=dict(arrowstyle="-|>", color=t["muted"], lw=1.5),
            )
    ax.text(
        0,
        -0.3,
        "steps 1–4 are batched, differentiable PyTorch ops; step 5 runs at zero "
        "temperature and gives an answer with no gap in it",
        fontsize=9.5,
        color=t["muted"],
        va="top",
    )
    ax.set_xlim(-0.05, 5 * W + 4 * gap + 0.05)
    ax.set_ylim(-0.6, 2.3)
    ax.axis("off")
    return fig


# ---- 1. a Kripke frame and its two operators -----------------------------------------


def fig_kripke(t):
    A, p = kripke_frame()
    box = F.necessity(p, A, tau=0.1)
    dia = F.possibility(p, A, tau=0.1)
    box_x = F.necessity(p, A, mode="exact")[:, 0]
    dia_x = F.possibility(p, A, mode="exact")[:, 0]
    pos = pentagon()
    r = 0.2
    panels = [
        (
            "p",
            "the valuation: true (filled)\nor false (hollow) in each world",
            p[:, 0],
            None,
        ),
        (
            "□p  (necessity)",
            "true where p holds in\nevery world you can reach",
            box_x,
            box,
        ),
        (
            "♢p  (possibility)",
            "true where p holds in\nsome world you can reach",
            dia_x,
            dia,
        ),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.4))
    for ax, (title, sub, crisp, soft) in zip(axes, panels):
        for a, b in KRIPKE_EDGES:
            if a == b:
                self_loop(ax, pos[a], t["muted"], r)
            else:
                arrow(ax, pos[a], pos[b], t["muted"], r=r, rad=0.12)
        for w, (x, y) in pos.items():
            on = crisp[w] > 0.5
            ax.add_patch(
                Circle(
                    (x, y),
                    r,
                    fc=t["accent"] if on else t["surface"],
                    ec=t["accent"] if on else t["axis"],
                    lw=2,
                    zorder=3,
                )
            )
            ax.text(
                x,
                y,
                f"w{w}",
                ha="center",
                va="center",
                fontsize=10,
                zorder=4,
                color="#ffffff" if on else t["ink2"],
                fontweight="bold",
            )
            if soft is not None:
                dy = -0.36 if y < 0.5 else 0.36
                ax.text(
                    x,
                    y + dy,
                    fmt(*soft[w]),
                    ha="center",
                    va="center",
                    fontsize=9,
                    color=t["ink2"],
                    zorder=4,
                )
        ax.set_xlim(-1.45, 1.45)
        ax.set_ylim(-1.35, 1.5)
        ax.set_aspect("equal")
        ax.axis("off")
        titled(ax, title, sub, t)
    fig.text(
        0.5,
        0.0,
        "grey labels: the soft interval [L, U] at τ = 0.1. "
        "Every one contains the crisp answer drawn by the fill.",
        ha="center",
        fontsize=9.5,
        color=t["muted"],
    )
    fig.subplots_adjust(wspace=0.08)
    return fig


# ---- 2. intervals that tighten with the temperature ----------------------------------


def fig_tau(t):
    A, p = kripke_frame()
    taus = torch.logspace(math.log10(0.004), 0, 60)
    box3 = torch.stack([F.necessity(p, A, tau=float(s))[3] for s in taus])
    dia2 = torch.stack([F.possibility(p, A, tau=float(s))[2] for s in taus])
    gap3 = torch.stack([F.box_width_entropy(A, p, tau=float(s))[3] for s in taus])
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.6), sharey=True)
    cases = [
        (
            axes[0],
            box3,
            1.0,
            "□p at w3  (crisp: true)",
            "w3 reaches w2 and itself, p holds in both",
        ),
        (
            axes[1],
            dia2,
            0.0,
            "♢p at w2  (crisp: false)",
            "w2 reaches only w4, where p fails",
        ),
    ]
    xs = taus.numpy()
    for ax, b, crisp, title, sub in cases:
        ax.fill_between(
            xs,
            b[:, 0].numpy(),
            b[:, 1].numpy(),
            color=t["accent_band"],
            lw=0,
            label="[L, U]: the soft interval",
        )
        ax.plot(xs, b[:, 0].numpy(), color=t["accent"], lw=1.6)
        ax.plot(xs, b[:, 1].numpy(), color=t["accent"], lw=1.6)
        ax.axhline(
            crisp,
            color=t["ink"],
            lw=1.2,
            ls=(0, (4, 3)),
            label="crisp modal-logic answer",
        )
        ax.set_xscale("log")
        ax.set_xlabel("temperature τ")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(axis="y", color=t["grid"], lw=0.8)
        ax.invert_xaxis()
        titled(ax, title, sub, t)
    keep = (1 - gap3) >= 0  # below 0 the lower bound is clipped
    axes[0].plot(
        xs[keep.numpy()],
        (1 - gap3[keep]).numpy(),
        "o",
        ms=3,
        color=t["ink2"],
        label="1 − box_width_entropy: the gap, predicted exactly",
    )
    axes[0].set_ylabel("truth value")
    h0, l0 = axes[0].get_legend_handles_labels()
    fig.legend(h0, l0, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.12))
    fig.subplots_adjust(wspace=0.08)
    return fig


# ---- 3. muddy children ---------------------------------------------------------------

MUDDY_WORLDS = list(product([0, 1], repeat=3))


def muddy_knows(agent, rnd, tau=0.05):
    alive = torch.tensor([float(sum(w) > rnd) for w in MUDDY_WORLDS])
    A = torch.tensor(
        [
            [
                float(all(w[k] == v[k] for k in range(3) if k != agent))
                for v in MUDDY_WORLDS
            ]
            for w in MUDDY_WORLDS
        ]
    )
    muddy = (
        torch.tensor([float(w[agent]) for w in MUDDY_WORLDS])
        .unsqueeze(-1)
        .expand(-1, 2)
    )
    return F.necessity(muddy, A * alive, tau=tau)[MUDDY_WORLDS.index((1, 1, 1))]


def fig_muddy(t):
    # isometric cube: bit 0 → right, bit 1 → up, bit 2 → depth
    def xy(w):
        return (w[0] * 1.0 + w[2] * 0.45, w[1] * 1.0 + w[2] * 0.38)

    child_col = [t["accent"], t["blue"], t["good"]]
    fig = plt.figure(figsize=(12.5, 3.9))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.25], wspace=0.25)
    subs = [
        "dad: “at least one of you\nis muddy” removes 000",
        "nobody answered: every\nworld with one muddy goes",
        "nobody answered again:\nonly 111 is left",
    ]
    for rnd in range(3):
        ax = fig.add_subplot(gs[rnd])
        alive = [sum(w) > rnd for w in MUDDY_WORLDS]
        for i, w in enumerate(MUDDY_WORLDS):
            for j, v in enumerate(MUDDY_WORLDS):
                diff = [k for k in range(3) if w[k] != v[k]]
                if i < j and len(diff) == 1:
                    both = alive[i] and alive[j]
                    (x0, y0), (x1, y1) = xy(w), xy(v)
                    ax.plot(
                        [x0, x1],
                        [y0, y1],
                        color=child_col[diff[0]] if both else t["grid"],
                        lw=2.2 if both else 1.2,
                        zorder=1,
                        solid_capstyle="round",
                    )
        for i, w in enumerate(MUDDY_WORLDS):
            x, y = xy(w)
            here = w == (1, 1, 1)
            ax.add_patch(
                Circle(
                    (x, y),
                    0.13,
                    zorder=3,
                    lw=1.6,
                    fc=(t["ink"] if here else t["box"]) if alive[i] else t["surface"],
                    ec=t["ink"] if alive[i] else t["grid"],
                )
            )
            ax.text(
                x,
                y - 0.23 if w[1] == 0 else y + 0.23,
                "".join(map(str, w)),
                ha="center",
                va="center",
                fontsize=8.5,
                color=t["ink2"] if alive[i] else t["axis"],
            )
        ax.set_xlim(-0.3, 1.75)
        ax.set_ylim(-0.4, 1.7)
        ax.set_aspect("equal")
        ax.axis("off")
        titled(ax, f"Round {rnd + 1}", subs[rnd], t)
    ax = fig.add_subplot(gs[3])
    for c in range(3):
        for rnd in range(3):
            lo, hi = muddy_knows(c, rnd)
            y = rnd * 4 + c
            ax.add_patch(Rectangle((0, y - 0.32), 1, 0.64, color=t["grid"], lw=0))
            ax.add_patch(
                Rectangle(
                    (float(lo), y - 0.32),
                    max(float(hi - lo), 0.012),
                    0.64,
                    color=child_col[c],
                    lw=0,
                )
            )
            ax.text(1.04, y, fmt(lo, hi), va="center", fontsize=8.5, color=t["ink2"])
    ax.set_yticks([1, 5, 9], ["round 1", "round 2", "round 3"])
    ax.set_xticks([0, 0.5, 1], ["0", "0.5", "1"])
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.8, 11)
    ax.invert_yaxis()
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    titled(
        ax,
        "Does child k know?",
        "K_k(muddy_k) in the real world 111,\none bar per child",
        t,
    )
    handles = [plt.Line2D([], [], color=c, lw=3) for c in child_col]
    fig.legend(
        handles,
        ["child 0 can't tell these apart", "child 1", "child 2"],
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.42, -0.06),
    )
    return fig


# ---- 4. learning the relation --------------------------------------------------------


def coloring_run(n=12, epochs=800, snaps=(0, 10, 800)):
    """Hide a graph, show only valid colourings, learn A from p_c → ¬♢p_c.

    Same code as section 4 of docs/visual-guide.md, plus snapshots.
    """
    g = torch.Generator().manual_seed(0)
    planted = torch.randint(0, 3, (n,), generator=g)
    hidden = torch.triu(
        (torch.rand(n, n, generator=g) < 0.2) & (planted[:, None] != planted), 1
    )
    hidden = (hidden | hidden.T).float()
    every = torch.cartesian_prod(*[torch.arange(3)] * n)
    clash = ((every[:, :, None] == every[:, None, :]) & hidden.bool()).any(-1).any(-1)
    valid = every[~clash][torch.randperm(int((~clash).sum()), generator=g)[:60]]
    P = torch.eye(3)[valid].transpose(1, 2).reshape(-1, n)

    logits = torch.nn.Parameter(torch.zeros(n, n))
    opt = torch.optim.Adam([logits], lr=0.05)
    out = {}
    for step in range(epochs + 1):
        A = torch.sigmoid(logits)
        A = 0.5 * (A + A.T) * (1 - torch.eye(n))
        if step in snaps:
            out[step] = A.detach().clone()
        dia = F.possibility(P, A.expand(len(P), n, n), tau=0.1)
        violation = (1 - F.implication(P, F.negation(dia))).sum() / len(valid)
        loss = violation - 2.0 * A.mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return hidden, out


def edge_auc(score, truth):
    iu = torch.triu_indices(*truth.shape, 1)
    s, y = score[iu[0], iu[1]], truth[iu[0], iu[1]]
    pos, neg = s[y == 1], s[y == 0]
    return float(
        (pos[:, None] > neg[None]).float().mean()
        + 0.5 * (pos[:, None] == neg[None]).float().mean()
    )


def fig_learn(t):
    truth, snaps = coloring_run()
    cmap = LinearSegmentedColormap.from_list("acc", t["seq"])
    fig, axes = plt.subplots(
        1, 5, figsize=(13, 3.4), gridspec_kw=dict(width_ratios=[1, 0.12, 1, 1, 1])
    )
    axes[1].axis("off")
    panels = [(axes[0], truth, "Hidden graph", "never shown to the model")]
    for ax, ep in zip(axes[2:], (0, 10, 800)):
        panels.append(
            (
                ax,
                snaps[ep],
                f"A_θ, epoch {ep}",
                f"edge-recovery AUC {edge_auc(snaps[ep], truth):.3f}",
            )
        )
    for ax, M, title, sub in panels:
        ax.imshow(M.numpy(), cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_color(t["border"])
        titled(ax, title, sub, t)
    fig.text(
        0.5,
        -0.02,
        "trained only on 60 valid 3-colourings, with the axiom  p_c → ¬♢p_c  "
        "(“no neighbour shares my colour”) as the loss",
        ha="center",
        fontsize=9.5,
        color=t["muted"],
    )
    fig.subplots_adjust(wspace=0.08)
    return fig


# ---- 5. the cost of nesting ----------------------------------------------------------


def fig_depth(t):
    A = torch.ones(8, 8)
    depth = 12
    taus = [
        (0.2, t["seq"][7]),
        (0.1, t["accent"]),
        (0.05, t["seq"][3]),
        (0.02, t["axis"]),
    ]
    fig, ax = plt.subplots(figsize=(8, 3.7))
    for tau, col in taus:
        b = torch.ones(8, 2)
        lows = [1.0]
        for _ in range(depth):
            b = F.necessity(b, A, tau=tau)
            lows.append(float(b[0, 0]))
        ax.plot(
            range(depth + 1), lows, "o-", color=col, lw=1.8, ms=4, label=f"τ = {tau}"
        )
    ax.axhline(0, color=t["ink2"], lw=1)
    ax.text(
        depth,
        0.12,
        "L = 0: the interval [0, 1] says nothing",
        ha="right",
        va="bottom",
        fontsize=9,
        color=t["ink2"],
    )
    ax.set_xlabel("nesting depth d   (□□…□ ⊤, 8 fully connected worlds)")
    ax.set_ylabel("lower bound L")
    ax.set_xticks(range(0, depth + 1, 2))
    ax.set_ylim(-0.06, 1.05)
    ax.grid(axis="y", color=t["grid"], lw=0.8)
    ax.legend(loc="upper right", ncol=2)
    titled(
        ax,
        "Every □ level costs τ·H(w) of width",
        "⊤ is true everywhere, so the crisp answer is 1 at every depth; "
        "U stays at 1 and L pays the gap",
        t,
    )
    return fig


# ---- 6. until over time --------------------------------------------------------------

UNTIL_PHI = [0.95, 0.9, 1.0, 0.85, 0.9, 0.95, 0.1, 0.9, 1.0, 0.95, 0.9, 0.8]
UNTIL_PSI = [0.0, 0.0, 0.05, 0.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.1, 0.95, 0.0]


def fig_until(t):
    T = len(UNTIL_PHI)
    phi = torch.tensor(UNTIL_PHI).unsqueeze(-1).expand(-1, 2)
    psi = torch.tensor(UNTIL_PSI).unsqueeze(-1).expand(-1, 2)
    u = F.until(phi, psi, torch.triu(torch.ones(T, T)))[:, 0]
    rows = [
        ("φ   “link is up”", UNTIL_PHI, t["blue"]),
        ("ψ   “ack received”", UNTIL_PSI, t["good"]),
        ("φ U ψ", u.tolist(), t["accent"]),
    ]
    fig, ax = plt.subplots(figsize=(11, 2.9))
    for r, (label, vals, col) in enumerate(rows):
        y = 2 - r
        for s, v in enumerate(vals):
            ax.add_patch(
                Rectangle((s + 0.06, y - 0.36), 0.88, 0.72, color=t["grid"], lw=0)
            )
            ax.add_patch(
                Rectangle((s + 0.06, y - 0.36), 0.88, 0.72 * v, color=col, lw=0)
            )
            ax.text(
                s + 0.5,
                y,
                f"{v:.2f}",
                ha="center",
                va="center",
                fontsize=8.5,
                color="#ffffff" if v > 0.55 else t["ink2"],
            )
        ax.text(-0.15, y, label, ha="right", va="center", fontsize=10.5, color=t["ink"])
    ax.set_xlim(-0.05, T)
    ax.set_ylim(-0.6, 2.6)
    ax.set_xticks([s + 0.5 for s in range(T)], [f"t{s}" for s in range(T)])
    ax.tick_params(axis="x", length=0)
    ax.set_yticks([])
    for s in ("left", "bottom"):
        ax.spines[s].set_visible(False)
    titled(
        ax,
        "Until: φ must hold at every step before ψ arrives",
        "the link drops at t6, so from t5 the t10 ack is out of reach; "
        "the backward sweep "
        "U_t = ψ_t ∨ (φ_t ∧ U_{t+1}) gets this in one pass",
        t,
    )
    return fig


# ---- 7. from a learned relation to a certificate -------------------------------------

CERT_EDGES = [
    (0, 1, 0.92),
    (1, 2, 0.85),
    (2, 7, 0.90),
    (0, 3, 0.60),
    (3, 4, 0.88),
    (4, 5, 0.90),
    (6, 6, 0.95),
    (5, 3, 0.70),
    (3, 6, 0.15),
    (6, 0, 0.30),
]
CERT_PHI = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0.2, 0.9], [0, 0], [1, 1]]


def cert_problem():
    A = torch.zeros(8, 8)
    for a, b, v in CERT_EDGES:
        A[a, b] = v
    return A, torch.tensor(CERT_PHI)


def fig_certify(t):
    A, phi = cert_problem()
    res = verify.round_and_certify(phi, A, operator="ef")
    pos = {
        0: (0, 1),
        1: (1.2, 1.7),
        2: (2.4, 1.7),
        7: (3.6, 1.2),
        3: (1.2, 0.3),
        4: (2.4, 0.3),
        5: (2.4, -0.8),
        6: (0.1, -0.6),
    }
    vcol = {"PROVEN": t["good"], "REFUTED": t["bad"], "UNDECIDED": t["unsure"]}
    r = 0.22
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    witness = res.witnesses[0]
    wit_edges = set(zip(witness, witness[1:]))
    for k, ax in enumerate(axes):
        rounded = k == 1
        for a, b, v in CERT_EDGES:
            keep = res.relation[a, b] > 0.5
            if rounded and not keep:
                continue
            col = t["accent"] if rounded and (a, b) in wit_edges else t["ink2"]
            lw = (
                2.6
                if rounded and (a, b) in wit_edges
                else (1.4 if rounded else 0.6 + 2.2 * v)
            )
            alpha = 1.0 if rounded else 0.25 + 0.75 * v
            if a == b:
                self_loop(ax, pos[a], col, r, lw=lw, alpha=alpha, up=False)
            else:
                arrow(ax, pos[a], pos[b], col, lw=lw, r=r, rad=0.15, alpha=alpha)
                if not rounded:
                    (x0, y0), (x1, y1) = pos[a], pos[b]
                    ax.text(
                        (x0 + x1) / 2 + 0.08 * (y1 - y0),
                        (y0 + y1) / 2 - 0.08 * (x1 - x0),
                        f"{v:.2f}",
                        fontsize=8,
                        color=t["ink2"],
                        ha="center",
                        va="center",
                        bbox=dict(fc=t["surface"], ec="none", pad=0.6),
                    )
        for w, (x, y) in pos.items():
            if rounded:
                fc = vcol[res.verdicts[w]]
                ec, ink = fc, "#ffffff"
            else:
                ec = (
                    t["accent"]
                    if phi[w, 0] > 0.5
                    else (t["unsure"] if phi[w, 1] > 0.5 else t["axis"])
                )
                fc, ink = t["surface"], t["ink"]
            ax.add_patch(Circle((x, y), r, fc=fc, ec=ec, lw=2, zorder=3))
            ax.text(
                x,
                y,
                f"w{w}",
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
                color=ink,
                zorder=4,
            )
            if not rounded and (phi[w, 1] > 0.5):
                ax.text(
                    x,
                    y + (0.32 if y > 0 else -0.32),
                    "goal = " + fmt(*phi[w]),
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    color=t["ink2"],
                )
        ax.set_xlim(-0.45, 4.1)
        ax.set_ylim(-1.35, 2.15)
        ax.set_aspect("equal")
        ax.axis("off")
    titled(
        axes[0],
        "Learned relation A_θ (graded)",
        "can the goal be reached from each world?  EF goal",
        t,
    )
    titled(
        axes[1],
        "Rounded frame, checked exactly",
        f"{res.n_flipped} edges flipped by rounding; witness for w0: "
        + " → ".join(f"w{w}" for w in witness),
        t,
    )
    handles = [Circle((0, 0), 1, color=c) for c in vcol.values()]
    fig.legend(
        handles,
        [
            "PROVEN: reachable on the rounded frame",
            "REFUTED: unreachable",
            "UNDECIDED: only via the uncertain goal w5",
        ],
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.04),
        handler_map={Circle: _CircleHandler()},
    )
    fig.subplots_adjust(wspace=0.05)
    return fig


class _CircleHandler:
    def legend_artist(self, legend, orig, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        h = handlebox.height
        p = Circle(
            (x0 + handlebox.width / 2, y0 + h / 2),
            h / 2,
            color=orig.get_facecolor(),
            transform=handlebox.get_transform(),
        )
        handlebox.add_artist(p)
        return p


FIGURES = {
    "pipeline": fig_pipeline,
    "kripke": fig_kripke,
    "tau": fig_tau,
    "muddy": fig_muddy,
    "learn": fig_learn,
    "depth": fig_depth,
    "until": fig_until,
    "certify": fig_certify,
}

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    only = sys.argv[1:] or list(FIGURES)
    for theme, th in THEMES.items():
        style(th)
        for name in only:
            save(FIGURES[name](th), name, theme)
            print("wrote", f"{name}-{theme}.svg")
