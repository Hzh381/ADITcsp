"""Compact ADiT architecture figure with module-level detail.

Visual grammar references:
- Kong et al., Efficient Frequency Domain-based Transformers for High-Quality
  Image Deblurring, CVPR 2023 (module symbols, grouped stages, detail blocks).
- Cheng et al., Effective Diffusion Transformer Architecture for Image
  Super-Resolution, arXiv:2409.19589 (small repeated transformer blocks).

Topology and shapes are ADiT-specific and come from the local code:
- src/models/encoders/transformer.py
- src/models/decoders/transformer.py
- src/models/vae_module.py
- src/models/ldm_module.py
- src/models/denoisers/dit.py
- configs/autoencoder_module/vae.yaml
- configs/diffusion_module/ldm.yaml
"""

from html import escape
from pathlib import Path

HERE = Path(__file__).resolve()
ROOT = HERE.parents[4]
OUT_DIR = ROOT / "ccfa-workfiles/figures/adit-architecture"
SVG_PATH = OUT_DIR / "source/figure.svg"

W, H = 1800, 1150
FONT = "Times New Roman"

INK = "#3C3A39"
MUTED = "#6B5F5A"
SOFT = "#8A7B73"
LINE = "#D9CEC8"
WHITE = "#FFFFFF"

BLUE = "#5B84B1"
BLUE_BG = "#F7FAFE"
BLUE_MOD = "#CCDFF2"
GREEN = "#2F7D5C"
GREEN_BG = "#E3F4EC"
PINK = "#B5486B"
PINK_BG = "#FDF7F9"
PINK_MOD = "#F7C9D9"
PEACH = "#B46A45"
PEACH_BG = "#FBF3EC"
PEACH_MOD = "#F3D8C4"
RED = "#B23A48"
RED_BG = "#FDECEF"

parts = []


def add(value):
    parts.append(value)


def rect(x, y, w, h, fill, stroke, rx=7, sw=1.2, dash=None):
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    add(
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{dash_attr}/>'
    )


def text(x, y, value, size=9, fill=INK, weight="normal", anchor="start"):
    add(
        f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" '
        f'fill="{fill}" font-weight="{weight}" text-anchor="{anchor}">{escape(value)}</text>'
    )


def line(x1, y1, x2, y2, color=SOFT, width=1.5, dash=None, marker=None):
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    marker_attr = f' marker-end="url(#{marker})"' if marker else ""
    add(
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
        f'stroke-width="{width}" stroke-linecap="round"{dash_attr}{marker_attr}/>'
    )


def path(d, color=SOFT, width=1.5, dash=None, marker=None):
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    marker_attr = f' marker-end="url(#{marker})"' if marker else ""
    add(
        f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}" '
        f'stroke-linecap="round" stroke-linejoin="round"{dash_attr}{marker_attr}/>'
    )


def module(x, y, w, h, title, lines=(), fill=BLUE_MOD, stroke=BLUE, ts=10, ls=8):
    rect(x, y, w, h, fill, stroke, rx=5, sw=1.0)
    title_size = min(ts, 8 if h < 30 else (9 if h < 40 else ts))
    text(x + 7, y + (12 if h < 30 else 14), title, size=title_size, weight="bold")
    line_bottom = y + h - 4
    line_step = 10
    for i, value in enumerate(lines):
        ly = line_bottom - (len(lines) - 1 - i) * line_step
        if ly <= y + (16 if h < 30 else 20):
            continue
        text(x + 7, ly, value, size=min(ls, 7.5), fill=MUTED)


def badge(x, y, w, h, value, fill=WHITE, stroke=LINE, color=MUTED, size=8):
    rect(x, y, w, h, fill, stroke, rx=5, sw=0.9)
    text(x + w / 2, y + h / 2 + 3, value, size=size, fill=color, anchor="middle")


add(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">')
add("<defs>")
for name, color in [
    ("aBlue", BLUE),
    ("aGreen", GREEN),
    ("aPink", PINK),
    ("aPeach", PEACH),
    ("aRed", RED),
    ("aGray", SOFT),
]:
    add(
        f'<marker id="{name}" viewBox="0 0 8 6" refX="7" refY="3" '
        f'markerWidth="6" markerHeight="6" orient="auto">'
        f'<path d="M0,0 L8,3 L0,6 z" fill="{color}"/></marker>'
    )
add("</defs>")
rect(0, 0, W, H, WHITE, WHITE, rx=0, sw=0)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
text(32, 42, "ADiT: All-atom Diffusion Transformer", size=28, weight="bold")
text(32, 66, "Stage 1 all-atom VAE  |  Stage 2 flow-matching DiT-B  |  module-level view", size=11, fill=MUTED)

ly = 38
line(1080, ly, 1114, ly, color=INK, width=1.8, marker="aGray")
text(1122, ly + 4, "forward", size=9, fill=MUTED)
line(1190, ly, 1224, ly, color=RED, width=1.5, dash="5 3", marker="aRed")
text(1232, ly + 4, "loss", size=9, fill=MUTED)
line(1290, ly, 1324, ly, color=SOFT, width=1.5, dash="2 3")
text(1332, ly + 4, "frozen", size=9, fill=MUTED)
rect(1410, ly - 7, 14, 14, BLUE_MOD, BLUE, rx=2, sw=0.8)
text(1430, ly + 4, "VAE layer", size=9, fill=MUTED)
rect(1530, ly - 7, 14, 14, PINK_MOD, PINK, rx=2, sw=0.8)
text(1550, ly + 4, "DiT block", size=9, fill=MUTED)
line(32, 80, 1768, 80, color="#E8DED8", width=0.8)


# ---------------------------------------------------------------------------
# Top-left: Stage 1 VAE pipeline
# ---------------------------------------------------------------------------
rect(32, 94, 820, 420, BLUE_BG, "#BBD5F0", rx=10, sw=1.2)
text(52, 120, "Stage 1 · VAE", size=17, weight="bold")
text(52, 138, "8+8 Transformer layers · enc 25.8M + dec 25.3M · latent 8", size=8.5, fill=MUTED)

# input
rect(52, 154, 150, 104, WHITE, "#8FB3D9", rx=6, sw=1.0)
text(64, 172, "All-atom sample", size=9, weight="bold")
text(64, 188, "atom_types [N]", size=8, fill=MUTED)
text(64, 202, "pos [N,3]", size=8, fill=MUTED)
text(64, 216, "frac [N,3]", size=8, fill=MUTED)
text(64, 230, "cell [B,3,3]", size=8, fill=MUTED)
text(64, 242, "lengths/angles [B,3]", size=7, fill=MUTED)
text(64, 254, "dataset_idx [B]", size=7, fill=MUTED)

# encoder container
rect(216, 154, 236, 248, "#E9F2FC", "#8FB3D9", rx=6, sw=1.0)
text(228, 170, "Encoder", size=10, weight="bold")
module(228, 180, 212, 26, "Atom Emb", ("Embedding 100→512",), BLUE_MOD, BLUE, 8, 7)
module(228, 212, 212, 26, "Pos MLP", ("Linear 3→512 · SiLU · Linear 512→512",), BLUE_MOD, BLUE, 8, 7)
module(228, 244, 212, 26, "Frac MLP", ("Linear 3→512 · SiLU · Linear 512→512",), BLUE_MOD, BLUE, 8, 7)
module(228, 276, 212, 24, "Token pos", ("sinusoidal index encoding",), WHITE, LINE, 8, 7)
module(228, 306, 212, 24, "Add", ("type + pos + frac + token",), WHITE, LINE, 8, 7)
module(228, 336, 212, 34, "Transformer ×8", ("MHSA 8 heads + FFN 512→2048→512",), BLUE_MOD, BLUE, 9, 7)
module(228, 376, 212, 22, "quant_conv", ("Linear 512→16",), GREEN_BG, GREEN, 8, 7)

# latent
rect(466, 228, 100, 98, GREEN_BG, GREEN, rx=6, sw=1.0)
text(480, 248, "Latent z", size=10, weight="bold")
text(480, 266, "z [B,N,8]", size=9, fill=GREEN)
text(480, 282, "μ, logσ²", size=8, fill=MUTED)
text(480, 298, "reparameterize", size=8, fill=MUTED)
text(480, 314, "sample z", size=8, fill=MUTED)

# decoder container
rect(590, 154, 236, 248, "#E9F2FC", "#8FB3D9", rx=6, sw=1.0)
text(602, 170, "Decoder", size=10, weight="bold")
module(602, 180, 212, 26, "post_quant + pos", ("Linear 8→512 · + token index",), GREEN_BG, GREEN, 8, 7)
module(602, 212, 212, 34, "Transformer ×8", ("MHSA 8 heads + FFN 512→2048→512",), BLUE_MOD, BLUE, 9, 7)
text(602, 258, "per-token heads", size=6.5, fill=SOFT)
text(710, 258, "pooled", size=6.5, fill=SOFT)
path("M 708 246 L 708 250 L 652 250 L 652 264", color=SOFT, width=1.1, marker="aGray")
path("M 708 250 L 762 250 L 762 264", color=SOFT, width=1.1, marker="aGray")
module(602, 264, 100, 22, "Atom", ("512→100",), WHITE, LINE, 7, 6.5)
module(602, 290, 100, 22, "Pos", ("512→3",), WHITE, LINE, 7, 6.5)
module(602, 316, 100, 22, "Frac", ("512→3",), WHITE, LINE, 7, 6.5)
module(710, 264, 104, 22, "Mean pool", ("over atoms",), WHITE, LINE, 7, 6.5)
module(710, 290, 104, 22, "Lattice", ("512→6",), WHITE, LINE, 7, 6.5)
module(710, 316, 104, 22, "PDB / CIF", (), WHITE, LINE, 7, 6.5)
text(602, 356, "atom/pos/frac are per-token;", size=6.5, fill=SOFT)
text(602, 368, "lattice reads the pooled vector", size=6.5, fill=SOFT)
text(602, 384, "shapes: [B,N,100] / [B,N,3] / [B,6]", size=6.2, fill=SOFT)

# VAE flow
line(202, 200, 216, 200, color=BLUE, marker="aBlue")
line(452, 250, 466, 250, color=BLUE, marker="aBlue")
path("M 566 250 L 580 250 L 580 193 L 590 193", color=GREEN, width=1.4, marker="aGreen")
text(470, 220, "z is the only bottleneck", size=7, fill=SOFT)

# output + loss
text(52, 424, "no skip connections; encoder and decoder are separate stacks", size=8, fill=SOFT)
rect(52, 434, 768, 44, RED_BG, RED, rx=6, sw=1.0, dash="5 3")
text(64, 451, "L_VAE = CE(atom types) + MSE(lengths/angles/frac/pos) + 1e-5·KL", size=8, fill=RED, weight="bold")
text(64, 466, "denoise: 10% atom-type mask + 10% coordinate noise", size=7.5, fill=RED)


# ---------------------------------------------------------------------------
# Top-right: Stage 2 DiT pipeline
# ---------------------------------------------------------------------------
rect(884, 94, 884, 420, PINK_BG, "#E8A9C0", rx=10, sw=1.2)
text(904, 120, "Stage 2 · DiT-B", size=17, weight="bold")
text(904, 138, "12 DiT blocks · d_model 768 · 12 heads · 129.7M", size=9, fill=MUTED)

module(904, 156, 150, 60, "Frozen encoder", ("z1 [B,N,8]", "frozen weights"), GREEN_BG, GREEN, 9, 8)
module(1074, 156, 170, 60, "Flow corruption", ("z_t=(1-t)z0+t·z1", "t ~ U(0.01,0.99)"), PEACH_MOD, PEACH, 9, 7.5)
module(1264, 156, 180, 60, "Token embed", ("x_t⊕x_sc [B,N,16]", "Linear 16→768 + pos"), BLUE_MOD, BLUE, 9, 7.5)
line(1054, 186, 1074, 186, color=PEACH, marker="aPeach")
line(1244, 186, 1264, 186, color=BLUE, marker="aBlue")

rect(1464, 148, 284, 82, WHITE, "#D8B4C4", rx=6, sw=1.0)
text(1476, 164, "Conditioning", size=9, weight="bold")
module(1472, 172, 84, 50, "t [B,1]", ("sinusoidal 256", "MLP 256→768→768"), PINK_BG, PINK, 7.5, 6.5)
module(1564, 172, 84, 50, "dataset", ("Emb(3,768)",), PINK_BG, PINK, 7.5, 6.5)
module(1656, 172, 84, 50, "spacegroup", ("Emb(231,768)", "off / null"), PINK_BG, PINK, 7.5, 6.5)
line(1514, 222, 1514, 230, color=PINK, width=1.2)
line(1606, 222, 1606, 230, color=PINK, width=1.2)
line(1698, 222, 1698, 230, color=PINK, width=1.2)
line(1514, 230, 1698, 230, color=PINK, width=1.3)
path("M 1606 230 L 1606 236 L 1300 236 L 1300 240", color=PINK, width=1.3, marker="aPink")
text(1306, 234, "c [B,768]", size=7, fill=PINK)

# DiT block grid
rect(904, 240, 470, 170, WHITE, "#D98CA8", rx=7, sw=1.0)
text(916, 258, "DiT-B × 12 blocks", size=10, weight="bold")
text(916, 272, "d_x 8 · d_model 768 · FFN 3072", size=7, fill=MUTED)
text(1150, 258, "S/B/L: 12/12/24", size=7, fill=PINK)
badge(1328, 248, 34, 18, "×12", PINK_BG, PINK, PINK, 7)
for row in range(3):
    for col in range(4):
        x = 916 + col * 110
        y = 282 + row * 40
        hi = row == 1 and col == 2
        rect(x, y, 100, 32, "#E89AB4" if hi else PINK_MOD, "#8E2F4E" if hi else PINK, rx=4, sw=1.2 if hi else 0.8)
        text(x + 50, y + 20, "DiT", size=7.5, fill=WHITE if hi else INK, anchor="middle")
        if hi:
            text(x + 50, y + 29, "detail", size=6, fill=WHITE, anchor="middle")

module(1394, 240, 160, 60, "Final layer", ("adaLN 768→1536", "Linear 768→8", "x1_hat [B,N,8]"), GREEN_BG, GREEN, 8, 6.5)
module(1394, 320, 160, 80, "Sampling", ("Euler 250 · CFG 4.0", "loop over DiT + final", "cond + null batch"), PEACH_MOD, PEACH, 8, 6.5)
module(1574, 320, 170, 80, "Frozen decoder", ("z [B,N,8]→atoms", "PDB / CIF"), GREEN_BG, GREEN, 9, 8)
line(1354, 216, 1354, 240, color=BLUE, marker="aBlue")
line(1254, 410, 1254, 240, color=SOFT, width=1.2, dash="3 2", marker="aGray")
path("M 1554 270 L 1564 270 L 1564 360 L 1574 360", color=SOFT, width=1.2, marker="aGray")

rect(904, 434, 844, 44, RED_BG, RED, rx=6, sw=1.0, dash="5 3")
text(916, 451, "L_FM = mean[mask · ((x1 - x1_hat)/(1-min(t,0.9)))²]", size=8, fill=RED, weight="bold")
text(916, 466, "mask = valid tokens; VAE frozen, only DiT updates", size=7.5, fill=RED)


# ---------------------------------------------------------------------------
# Bottom-left: VAE Transformer layer detail
# ---------------------------------------------------------------------------
rect(32, 536, 820, 580, BLUE_BG, "#BBD5F0", rx=10, sw=1.2)
text(52, 562, "VAE Transformer layer", size=17, weight="bold")
text(52, 580, "shared by encoder and decoder · ×8 each · pre-norm residual", size=9, fill=MUTED)

module(52, 600, 170, 44, "Input", ("x [B,N,512]",), WHITE, LINE, 9, 8)
module(52, 660, 170, 30, "LayerNorm", ("512",), WHITE, LINE, 8, 7)
line(137, 644, 137, 660, color=SOFT, marker="aGray")

rect(240, 600, 560, 170, WHITE, "#8FB3D9", rx=6, sw=1.0)
text(254, 618, "Multi-head self-attention (8 heads, d=64)", size=9, weight="bold")
module(260, 630, 220, 28, "QKV", ("Linear 512→1536",), BLUE_MOD, BLUE, 8, 7)
module(260, 666, 220, 28, "Split heads", ("8 heads × 64",), BLUE_MOD, BLUE, 8, 7)
module(500, 630, 280, 28, "softmax", ("QKᵀ / √64",), WHITE, LINE, 8, 7)
module(500, 666, 280, 28, "Weighted V", ("·V → concat",), WHITE, LINE, 8, 7)
module(380, 704, 280, 28, "Output projection", ("Linear 512→512",), BLUE_MOD, BLUE, 8, 7)
path("M 470 644 L 500 644", color=SOFT, marker="aGray")
path("M 470 680 L 500 680", color=SOFT, marker="aGray")
path("M 640 694 L 640 704", color=SOFT, marker="aGray")
path("M 222 675 L 240 675", color=BLUE, marker="aBlue")

module(52, 790, 170, 30, "Add + residual", ("x + attn",), WHITE, LINE, 8, 7)
module(52, 850, 170, 30, "LayerNorm", ("512",), WHITE, LINE, 8, 7)
line(660, 732, 660, 770, color=SOFT, marker="aGray")
path("M 660 770 L 137 770 L 137 790", color=SOFT, marker="aGray")
line(137, 820, 137, 850, color=SOFT, marker="aGray")
path("M 222 865 L 240 865", color=SOFT, marker="aGray")

rect(240, 810, 560, 100, WHITE, "#8FB3D9", rx=6, sw=1.0)
text(254, 828, "Feed-forward network", size=9, weight="bold")
module(260, 840, 160, 26, "Linear", ("512→2048",), BLUE_MOD, BLUE, 8, 7)
module(430, 840, 90, 26, "GELU", (), WHITE, LINE, 8, 7)
module(530, 840, 160, 26, "Linear", ("2048→512",), BLUE_MOD, BLUE, 8, 7)
path("M 420 853 L 430 853", color=SOFT, marker="aGray")
path("M 520 853 L 530 853", color=SOFT, marker="aGray")

module(52, 930, 170, 30, "Add + residual", ("x + FFN",), WHITE, LINE, 8, 7)
module(52, 990, 170, 44, "Output", ("x [B,N,512]",), WHITE, LINE, 9, 8)
path("M 660 910 L 660 920 L 137 920 L 137 930", color=SOFT, marker="aGray")
line(137, 960, 137, 990, color=SOFT, marker="aGray")
badge(740, 980, 44, 20, "×8", BLUE_BG, BLUE, BLUE, 8)
text(52, 1060, "dropout = 0 · norm_first = True · residual per sublayer", size=8, fill=SOFT)
text(52, 1074, "decoder uses the same non-causal TransformerEncoder stack", size=7.5, fill=SOFT)


# ---------------------------------------------------------------------------
# Bottom-right: DiT block detail
# ---------------------------------------------------------------------------
rect(884, 536, 884, 580, PINK_BG, "#E8A9C0", rx=10, sw=1.2)
text(904, 562, "DiT block", size=17, weight="bold")
text(904, 580, "one of 12 blocks · ≈10.63M/block · adaLN conditioning · ×12", size=8.5, fill=MUTED)

module(904, 600, 180, 50, "Input", ("x [B,N,768]", "c [B,768]"), WHITE, LINE, 9, 8)
module(1100, 600, 300, 70, "adaLN", ("SiLU → Linear 768→4608", "6×768: γ1,β1,α1,γ2,β2,α2", "zero-init gates"), PEACH_MOD, PEACH, 9, 7)
module(904, 680, 180, 30, "LayerNorm", ("768",), WHITE, LINE, 8, 7)
line(994, 650, 994, 680, color=SOFT, marker="aGray")

rect(1100, 680, 620, 150, WHITE, "#D98CA8", rx=6, sw=1.0)
text(1114, 698, "Multi-head self-attention (12 heads, d=64)", size=9, weight="bold")
module(1120, 710, 220, 28, "QKV", ("Linear 768→2304",), PINK_MOD, PINK, 8, 7)
module(1120, 746, 220, 28, "Split heads", ("12 heads × 64",), PINK_MOD, PINK, 8, 7)
module(1360, 710, 340, 28, "softmax", ("QKᵀ / √64",), WHITE, LINE, 8, 7)
module(1360, 746, 340, 28, "Weighted V", ("·V → concat",), WHITE, LINE, 8, 7)
module(1240, 790, 280, 28, "Output projection", ("Linear 768→768",), PINK_MOD, PINK, 8, 7)
path("M 1340 724 L 1360 724", color=SOFT, marker="aGray")
path("M 1340 760 L 1360 760", color=SOFT, marker="aGray")
path("M 1530 774 L 1530 790", color=SOFT, marker="aGray")
path("M 1084 695 L 1100 695", color=PINK, marker="aPink")

module(904, 850, 180, 30, "Gate + residual", ("α1 ⊙ + x",), PEACH_MOD, PEACH, 8, 7)
module(904, 910, 180, 30, "LayerNorm", ("768",), WHITE, LINE, 8, 7)
path("M 1520 804 L 1540 804 L 1540 870 L 1084 870", color=SOFT, marker="aGray")
line(994, 880, 994, 910, color=SOFT, marker="aGray")

rect(1100, 900, 620, 90, WHITE, "#D98CA8", rx=6, sw=1.0)
text(1114, 918, "Feed-forward network", size=9, weight="bold")
module(1120, 930, 180, 26, "Linear", ("768→3072",), PINK_MOD, PINK, 8, 7)
module(1320, 930, 90, 26, "GELU", (), WHITE, LINE, 8, 7)
module(1430, 930, 180, 26, "Linear", ("3072→768",), PINK_MOD, PINK, 8, 7)
path("M 1300 943 L 1320 943", color=SOFT, marker="aGray")
path("M 1410 943 L 1430 943", color=SOFT, marker="aGray")

module(904, 1010, 180, 30, "Gate + residual", ("α2 ⊙ + x",), PEACH_MOD, PEACH, 8, 7)
module(904, 1070, 180, 44, "Output", ("x [B,N,768]", "to next block"), WHITE, LINE, 9, 8)
line(994, 940, 994, 1010, color=SOFT, marker="aGray")
path("M 1610 943 L 1630 943 L 1630 1030 L 1084 1030", color=SOFT, marker="aGray")
line(994, 1040, 994, 1070, color=SOFT, marker="aGray")

# adaLN bus to gates
path("M 1100 650 L 1092 650 L 1092 870 L 1084 870", color=PEACH, width=1.3, marker="aPeach")
path("M 1092 870 L 1092 1030 L 1084 1030", color=PEACH, width=1.3, marker="aPeach")
text(1044, 900, "γ,β,α", size=7, fill=PEACH)

rect(1320, 1010, 400, 60, WHITE, LINE, rx=5, sw=0.8)
text(1332, 1026, "x ← x + α1⊙MHA(LN(x)·(1+γ1)+β1)", size=7.5)
text(1332, 1042, "x ← x + α2⊙MLP(LN(x)·(1+γ2)+β2)", size=7.5)
text(1332, 1058, "c = t + d + s · mask padded tokens", size=7, fill=SOFT)
badge(1600, 1078, 44, 20, "×12", PINK_BG, PINK, PINK, 8)
text(1320, 1098, "2 dataset classes + null · 230 SG + null", size=6.5, fill=SOFT)
text(1320, 1110, "class dropout 0.1 · spacegroup off in released checkpoint", size=6.2, fill=SOFT)


add("</svg>")
SVG_PATH.parent.mkdir(parents=True, exist_ok=True)
SVG_PATH.write_text("\n".join(parts), encoding="utf-8")
print(SVG_PATH)
