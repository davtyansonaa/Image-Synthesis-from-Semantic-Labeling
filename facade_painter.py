import tkinter as tk
import tkinter.messagebox
from tkinter.filedialog import asksaveasfilename
import threading
import numpy as np
from PIL import Image, ImageTk
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


# ── Model ─────────────────────────────────────────────────────────────────────

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=4, stride=2, padding=1,
                      bias=not use_batchnorm)
        ]
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_dropout=False):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels,
                               kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        ]
        if use_dropout:
            layers.append(nn.Dropout(0.5))
        layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x, skip):
        x = self.block(x)
        return torch.cat([x, skip], dim=1)


class UNetGenerator(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, ngf=64, depth=8, dropout=0.5):
        super().__init__()
        assert depth >= 4
        self.depth = depth
        self.encoders = nn.ModuleList()
        encoder_channels = []
        in_ch = input_nc
        for i in range(depth):
            if i == 0:
                out_ch = ngf;          use_bn = False
            elif i == depth - 1:
                out_ch = ngf * 8;      use_bn = False
            elif i <= 3:
                out_ch = ngf * min(2 ** i, 8); use_bn = True
            else:
                out_ch = ngf * 8;      use_bn = True
            self.encoders.append(EncoderBlock(in_ch, out_ch, use_batchnorm=use_bn))
            encoder_channels.append(out_ch)
            in_ch = out_ch
        self.decoders = nn.ModuleList()
        in_ch = encoder_channels[-1]
        for i in range(depth - 1):
            enc_idx  = depth - 2 - i
            skip_ch  = encoder_channels[enc_idx]
            use_drop = (i < 3) and (dropout > 0)
            out_ch   = skip_ch
            self.decoders.append(DecoderBlock(in_ch, out_ch, use_dropout=use_drop))
            in_ch = out_ch + skip_ch
        self.final = nn.Sequential(
            nn.ConvTranspose2d(in_ch, output_nc, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        encoder_features = []
        h = x
        for encoder in self.encoders:
            h = encoder(h)
            encoder_features.append(h)
        h = encoder_features[-1]
        for i, decoder in enumerate(self.decoders):
            skip = encoder_features[self.depth - 2 - i]
            h = decoder(h, skip)
        return self.final(h)


# ── Constants ──────────────────────────────────────────────────────────────────

IMAGE_SIZE  = 128
UNET_DEPTH  = 7
NGF         = 64
DROPOUT     = 0.5
CANVAS_SIZE = 512

PALETTE = {
    "Wall":       "#C0C0C0",
    "Window":     "#4A90D9",
    "Door":       "#8B4513",
    "Balcony":    "#F5A623",
    "Cornice":    "#7B68EE",
    "Sill":       "#50C878",
    "Pillar":     "#FF6B6B",
    "Shop":       "#FFD700",
    "Blind":      "#FF8C00",
    "Deco":       "#DA70D6",
    "Molding":    "#20B2AA",
    "Background": "#2C3E50",
}


# ── Colour helpers ─────────────────────────────────────────────────────────────

def _brighten(hex_col, amount=22):
    r = min(255, int(hex_col[1:3], 16) + amount)
    g = min(255, int(hex_col[3:5], 16) + amount)
    b = min(255, int(hex_col[5:7], 16) + amount)
    return f"#{r:02x}{g:02x}{b:02x}"


def _contrast(hex_col):
    r, g, b = int(hex_col[1:3], 16), int(hex_col[3:5], 16), int(hex_col[5:7], 16)
    return "#000000" if (r * 299 + g * 587 + b * 114) / 1000 > 128 else "#FFFFFF"


# ── Canvas-based button ────────────────────────────────────────────────────────
#
#  WHY Canvas and not Button/Label?
#  On macOS the Aqua theme overrides bg on tk.Button entirely.
#  tk.Label also ignores bg for many colour values on the system Tk.
#  tk.Canvas ALWAYS renders its own bg colour on every platform.  ✓

class CanvasButton(tk.Canvas):
    """Reliable coloured button for macOS (and every other platform)."""

    def __init__(self, parent, text, bg,
                 fg=None,
                 font=("Courier New", 9),
                 btn_width=148, btn_height=30,
                 cursor="hand2", command=None, **kw):

        self._bg      = bg
        self._hover   = _brighten(bg)
        self._fg      = fg if fg else _contrast(bg)
        self._cmd     = command
        self._selected = False

        super().__init__(
            parent,
            width=btn_width, height=btn_height,
            bg=bg,
            highlightthickness=3,
            highlightbackground=bg,   # invisible border (same as bg)
            cursor=cursor,
            **kw,
        )

        cx, cy = btn_width // 2, btn_height // 2
        self._txt = self.create_text(cx, cy, text=text,
                                     fill=self._fg, font=font, anchor="center")

        self.bind("<Enter>",    self._enter)
        self.bind("<Leave>",    self._leave)
        self.bind("<Button-1>", self._click)
        self.tag_bind(self._txt, "<Button-1>", self._click)

    def _enter(self, _e):
        if not self._selected:
            self.config(bg=self._hover)

    def _leave(self, _e):
        if not self._selected:
            self.config(bg=self._bg)

    def _click(self, _e):
        if self._cmd:
            self._cmd()

    def set_selected(self, on: bool):
        """Show a white border when selected; invisible border when not."""
        self._selected = on
        self.config(
            bg=self._hover if on else self._bg,
            highlightbackground="#FFFFFF" if on else self._bg,
        )


# ── Main application ───────────────────────────────────────────────────────────

class FacadePainter:
    DARK   = "#1A1A2E"
    PANEL  = "#16213E"
    ACCENT = "#E94560"
    TEXT   = "#EAEAEA"

    def __init__(self, root, generator, device):
        self.root      = root
        self.generator = generator
        self.device    = device
        self.brush_size    = 24
        self.current_color = "#C0C0C0"
        self.current_label = "Wall"
        self.last_output   = None

        root.title("Facade Painter · pix2pix")
        root.configure(bg=self.DARK)
        root.resizable(False, False)

        self.canvas_array = np.full((CANVAS_SIZE, CANVAS_SIZE, 3), 44, dtype=np.uint8)
        self._build_ui()
        self._redraw_canvas()

    # ── UI ─────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        tk.Label(self.root, text="FACADE PAINTER",
                 font=("Courier New", 14, "bold"),
                 bg=self.DARK, fg=self.ACCENT).pack(pady=(12, 2))
        tk.Label(self.root, text="paint a label map  →  click Generate",
                 font=("Courier New", 9), bg=self.DARK, fg="#555").pack(pady=(0, 10))

        main = tk.Frame(self.root, bg=self.DARK)
        main.pack(padx=14, pady=4)

        # ── Left: palette ──────────────────────────────────────────────────────
        pf = tk.Frame(main, bg=self.PANEL, padx=8, pady=8)
        pf.grid(row=0, column=0, padx=(0, 10), sticky="n")

        tk.Label(pf, text="LABELS", font=("Courier New", 9, "bold"),
                 bg=self.PANEL, fg=self.ACCENT).pack(pady=(0, 6))

        self.palette_btns: dict[str, CanvasButton] = {}
        for label, color in PALETTE.items():
            btn = CanvasButton(
                pf,
                text       = f"  {label:<12}",
                bg         = color,
                font       = ("Courier New", 9),
                btn_width  = 148,
                btn_height = 30,
                command    = (lambda c=color, l=label: self._pick(c, l)),
            )
            btn.pack(pady=2)
            self.palette_btns[label] = btn

        tk.Label(pf, text="BRUSH SIZE", font=("Courier New", 8),
                 bg=self.PANEL, fg="#888").pack(pady=(12, 0))
        self.brush_var = tk.IntVar(value=self.brush_size)
        tk.Scale(pf, from_=4, to=100, orient="horizontal",
                 variable=self.brush_var, bg=self.PANEL, fg=self.TEXT,
                 troughcolor=self.DARK, highlightthickness=0, bd=0,
                 command=lambda v: setattr(self, "brush_size", int(v))
                 ).pack(fill="x")

        # ── Centre: drawing canvas ─────────────────────────────────────────────
        cf = tk.Frame(main, bg=self.PANEL, padx=4, pady=4)
        cf.grid(row=0, column=1, padx=(0, 10))

        self.indicator = tk.Canvas(cf, width=CANVAS_SIZE, height=24,
                                   bg=self.PANEL, highlightthickness=0)
        self.indicator.pack()
        self._draw_indicator()

        self.canvas = tk.Canvas(cf, width=CANVAS_SIZE, height=CANVAS_SIZE,
                                bg="#2C2C2C", cursor="crosshair",
                                highlightthickness=2,
                                highlightbackground=self.ACCENT)
        self.canvas.pack()
        self.canvas.bind("<ButtonPress-1>", self._press)
        self.canvas.bind("<B1-Motion>",     self._drag)

        # ── Right: output ──────────────────────────────────────────────────────
        of = tk.Frame(main, bg=self.PANEL, padx=4, pady=4)
        of.grid(row=0, column=2, sticky="n")

        tk.Label(of, text="GENERATED", font=("Courier New", 9, "bold"),
                 bg=self.PANEL, fg=self.ACCENT).pack(pady=(0, 4))
        tk.Canvas(of, width=CANVAS_SIZE, height=24,
                  bg=self.PANEL, highlightthickness=0).pack()

        self.out_canvas = tk.Canvas(of, width=CANVAS_SIZE, height=CANVAS_SIZE,
                                    bg="#0D0D1A", highlightthickness=1,
                                    highlightbackground="#333")
        self.out_canvas.pack()
        self.out_canvas.create_text(CANVAS_SIZE // 2, CANVAS_SIZE // 2,
                                    text="press  [ GENERATE ]",
                                    fill="#444", font=("Courier New", 11))

        # ── Bottom: action buttons ─────────────────────────────────────────────
        br = tk.Frame(self.root, bg=self.DARK)
        br.pack(pady=10)

        for text, cmd, bg in [
            ("⚡  GENERATE", self._generate, self.ACCENT),
            ("✕  CLEAR",    self._clear,    "#3a3a5a"),
            ("⬇  SAVE",     self._save,     "#3a3a5a"),
        ]:
            CanvasButton(br, text=text, bg=bg, fg=self.TEXT,
                         font=("Courier New", 11, "bold"),
                         btn_width=170, btn_height=36,
                         command=cmd).pack(side="left", padx=5)

        self.status = tk.StringVar(value="Select a label and start painting.")
        tk.Label(self.root, textvariable=self.status,
                 font=("Courier New", 8), bg=self.DARK, fg="#555").pack(pady=(0, 8))

        self._pick("#C0C0C0", "Wall")

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _hex_to_rgb(self, h):
        return tuple(int(h[i:i+2], 16) for i in (1, 3, 5))

    def _pick(self, color, label):
        self.current_color = color
        self.current_label = label
        self.status.set(f"Painting: {label}  ({color})")
        self._draw_indicator()
        for l, btn in self.palette_btns.items():
            btn.set_selected(l == label)

    def _draw_indicator(self):
        self.indicator.delete("all")
        self.indicator.create_rectangle(0, 0, CANVAS_SIZE, 24,
                                        fill=self.current_color, outline="")
        self.indicator.create_text(CANVAS_SIZE // 2, 12,
                                   text=self.current_label,
                                   fill=_contrast(self.current_color),
                                   font=("Courier New", 9, "bold"))

    # ── Painting ───────────────────────────────────────────────────────────────

    def _paint(self, x, y):
        r  = self.brush_size // 2
        x0 = max(0, x - r);           y0 = max(0, y - r)
        x1 = min(CANVAS_SIZE, x + r); y1 = min(CANVAS_SIZE, y + r)
        self.canvas_array[y0:y1, x0:x1] = self._hex_to_rgb(self.current_color)
        self.canvas.create_rectangle(x0, y0, x1, y1,
                                     fill=self.current_color, outline="")

    def _press(self, e): self._paint(e.x, e.y)
    def _drag(self, e):  self._paint(e.x, e.y)

    def _redraw_canvas(self):
        img = Image.fromarray(self.canvas_array, "RGB")
        self._tk_draw = ImageTk.PhotoImage(img)
        self.canvas.create_image(0, 0, anchor="nw", image=self._tk_draw)

    def _clear(self):
        self.canvas_array[:] = 44
        self.canvas.delete("all")
        self._redraw_canvas()
        self.out_canvas.delete("all")
        self.out_canvas.create_text(CANVAS_SIZE // 2, CANVAS_SIZE // 2,
                                    text="press  [ GENERATE ]",
                                    fill="#444", font=("Courier New", 11))
        self.last_output = None
        self.status.set("Canvas cleared.")

    # ── Inference ──────────────────────────────────────────────────────────────

    def _generate(self):
        self.status.set("Generating…")
        self.root.update()
        threading.Thread(target=self._infer, daemon=True).start()

    def _infer(self):
        try:
            img = Image.fromarray(self.canvas_array, "RGB")
            tf  = transforms.Compose([
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize([0.5] * 3, [0.5] * 3),
            ])
            tensor = tf(img).unsqueeze(0).to(self.device)
            self.generator.eval()
            with torch.no_grad():
                out = self.generator(tensor)
            out_np = out[0].cpu().permute(1, 2, 0).numpy()
            out_np = ((out_np + 1.0) / 2.0 * 255).clip(0, 255).astype(np.uint8)
            self.last_output = Image.fromarray(out_np).resize(
                (CANVAS_SIZE, CANVAS_SIZE), Image.BILINEAR)
            self.root.after(0, self._show_output)
        except Exception as ex:
            self.root.after(0, lambda: self.status.set(f"Error: {ex}"))

    def _show_output(self):
        self._tk_out = ImageTk.PhotoImage(self.last_output)
        self.out_canvas.delete("all")
        self.out_canvas.create_image(0, 0, anchor="nw", image=self._tk_out)
        self.status.set("Done!  Draw more or save the result.")

    # ── Save ───────────────────────────────────────────────────────────────────

    def _save(self):
        if self.last_output is None:
            tk.messagebox.showinfo("Nothing to save", "Generate an image first.")
            return
        path = asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")],
            initialfile="facade_result.png",
        )
        if path:
            label_img = Image.fromarray(self.canvas_array, "RGB")
            combined  = Image.new("RGB", (CANVAS_SIZE * 2 + 8, CANVAS_SIZE), (20, 20, 20))
            combined.paste(label_img,        (0, 0))
            combined.paste(self.last_output, (CANVAS_SIZE + 8, 0))
            combined.save(path)
            self.status.set(f"Saved → {path}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="outputs/checkpoints/latest.pth")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    )
    G = UNetGenerator(input_nc=3, output_nc=3, ngf=NGF,
                      depth=UNET_DEPTH, dropout=DROPOUT).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    G.load_state_dict(ckpt["generator_state_dict"])
    G.eval()

    root = tk.Tk()
    FacadePainter(root, G, device)
    root.mainloop()


if __name__ == "__main__":
    main()