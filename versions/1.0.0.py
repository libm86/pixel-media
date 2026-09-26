"""Pixel Müzik - Material You 3 esintili, çevrimdışı müzik çalar (CustomTkinter)

Kurulum:  pip install customtkinter pillow numpy sounddevice soundfile mutagen syncedlyrics av
Formatlar: mp3, flac, wav, ogg, m4a, aac, opus, wma... (PyAV/FFmpeg)   |   Boşluk tuşu: oynat/duraklat
"""
import bisect, colorsys, io, json, math, os, queue, random, re, threading, time
import numpy as np
import customtkinter as ctk
import sounddevice as sd
import soundfile as sf
from tkinter import Canvas, filedialog
from mutagen import File as MFile
from PIL import Image, ImageDraw, ImageFilter, ImageOps

try:
    import syncedlyrics
except ImportError:
    syncedlyrics = None

EXTS = (".mp3", ".flac", ".wav", ".ogg", ".opus", ".m4a", ".aac", ".wma", ".aiff", ".aif", ".alac", ".mka", ".webm", ".mp4")
SF_EXTS = (".wav", ".flac", ".ogg", ".mp3", ".aiff", ".aif")
CFG = os.path.join(os.path.expanduser("~"), ".pixel_muzik.json")
DEF = dict(folder="", theme="Açık", dyn=True, bg="Bulanık", radius=28, xf=4, hifi=True, online=True, vol=0.8)
SEED = (103, 80, 164)
ART = 320


# ---------- Renk (Material You: tohum renkten ton paleti) ----------
def hexc(h, s, l):
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return "#%02x%02x%02x" % (int(r * 255), int(g * 255), int(b * 255))

def lerp(a, b, k):
    A = [int(a[i:i + 2], 16) for i in (1, 3, 5)]
    B = [int(b[i:i + 2], 16) for i in (1, 3, 5)]
    return "#%02x%02x%02x" % tuple(int(x + (y - x) * k) for x, y in zip(A, B))

def dominant(img):
    q = img.resize((64, 64)).quantize(6).convert("RGB")
    best, score = SEED, -1
    for n, c in q.getcolors(64 * 64):
        _, l, s = colorsys.rgb_to_hls(*[v / 255 for v in c])
        sc = n * (0.15 + s) * (1 - abs(l - 0.5))
        if sc > score:
            best, score = c, sc
    return best

def palette(rgb, dark):
    h, _, s = colorsys.rgb_to_hls(*[v / 255 for v in rgb])
    s = min(max(s, 0.35), 0.75)
    t = (dict(primary=.80, on_primary=.20, container=.30, on_container=.90, surface=.07, card=.12, on_surface=.92, variant=.72)
         if dark else
         dict(primary=.40, on_primary=.99, container=.90, on_container=.12, surface=.93, card=.985, on_surface=.10, variant=.40))
    return {k: hexc(h, s * (.25 if k in ("surface", "card") else 1), v) for k, v in t.items()}


# ---------- Görsel yardımcılar ----------
def rounded(img, size, r):
    """Yumuşak köşe: 4x süper örneklenmiş maske."""
    img = ImageOps.fit(img, (size, size), Image.LANCZOS).convert("RGBA")
    m = Image.new("L", (size * 4, size * 4), 0)
    ImageDraw.Draw(m).rounded_rectangle((0, 0, size * 4 - 1, size * 4 - 1), min(r, size // 2) * 4, fill=255)
    img.putalpha(m.resize((size, size), Image.LANCZOS))
    return img

def gen_cover(seed):
    rnd = random.Random(seed); h = rnd.random()
    x = np.linspace(0, 1, 256); g = ((x[None, :] + x[:, None]) / 2)[..., None]
    c1 = np.array(colorsys.hls_to_rgb(h, .45, .7)) * 255
    c2 = np.array(colorsys.hls_to_rgb((h + .15) % 1, .62, .7)) * 255
    return Image.fromarray((c1 * (1 - g) + c2 * g).astype("uint8"))

def swirl(im, covers, r, a=.55):
    """Kolaj deseni (Cosmic Swirl): 3 kollu galaksi sarmalı; kapaklar dışa doğru büyür."""
    w, h = im.size; m = min(w, h); base = m * .24
    out = im.convert("RGBA")
    for i in range(35, -1, -1):
        arm, j = i % 3, i // 3
        ang = arm * 2.094 + j * .5; d = m * (.07 + .04 * j); sz = int(base * (.5 + .05 * j))
        tile = rounded(covers[i % len(covers)], sz * 2, r * 2).rotate(math.degrees(ang) + 90, expand=True, resample=Image.BICUBIC)
        tile = tile.resize((tile.width // 2, tile.height // 2), Image.LANCZOS)        # kenarları yumuşat
        tile.putalpha(tile.getchannel("A").point(lambda v: int(v * a)))
        out.paste(tile, (int(w / 2 + d * math.cos(ang) - tile.width / 2), int(h / 2 + d * math.sin(ang) - tile.height / 2)), tile)
    return out.convert("RGB")


def make_bg(cover, covers, w, h, mode, c, R, rects):
    """Bulanık kapak (+ Cosmic Swirl) üstüne yumuşak gölgeli, yuvarlak köşeli kartları çizer."""
    s = 2; w2, h2 = max(w // s, 8), max(h // s, 8)
    im = ImageOps.fit(cover, (w2, h2)).filter(ImageFilter.GaussianBlur(22))      # bulanıklık efekti
    tint = Image.new("RGB", (w2, h2), c["surface"])
    im = Image.blend(im, tint, .72)
    if mode == "Cosmic Swirl": im = Image.blend(swirl(im, covers, R // s), tint, .25)
    im = im.resize((w, h), Image.BICUBIC)
    sh = Image.new("L", (w, h), 0); d = ImageDraw.Draw(sh)
    for x, y, cw, ch in rects: d.rounded_rectangle((x, y + 8, x + cw, y + ch + 8), R, fill=255)
    sh = sh.filter(ImageFilter.GaussianBlur(16)).point(lambda v: int(v * .25))
    im.paste(Image.new("RGB", (w, h), c["on_surface"]), (0, 0), sh)
    for x, y, cw, ch in rects:
        m = Image.new("L", (cw * 3, ch * 3), 0)                                    # 3x süper örnekleme
        ImageDraw.Draw(m).rounded_rectangle((0, 0, cw * 3 - 1, ch * 3 - 1), R * 3, fill=255)
        im.paste(Image.new("RGB", (cw, ch), c["card"]), (x, y), m.resize((cw, ch), Image.LANCZOS))
    return im


# ---------- Etiket, söz, ses ----------
def meta(path):
    t = dict(path=path, title=os.path.splitext(os.path.basename(path))[0], artist="Bilinmeyen sanatçı", cover=None)
    try:
        m = MFile(path); e = MFile(path, easy=True)
        if e:
            t["title"] = e.get("title", [t["title"]])[0]; t["artist"] = e.get("artist", [t["artist"]])[0]
        raw = m.pictures[0].data if getattr(m, "pictures", None) else None
        if raw is None and m.tags:
            raw = next((m.tags[k].data for k in m.tags.keys() if k.startswith("APIC")), None)
        if raw is None and m.tags and "covr" in m.tags: raw = bytes(m.tags["covr"][0])   # m4a kapağı
        if raw:
            t["cover"] = Image.open(io.BytesIO(raw)).convert("RGB"); t["cover"].thumbnail((320, 320))
    except Exception:
        pass
    t["cover"] = t["cover"] or gen_cover(t["title"])
    return t

def parse_lrc(text):
    L = []
    for ln in text.splitlines():
        stamps = re.findall(r"\[(\d+):(\d+(?:\.\d+)?)\]", ln)
        txt = re.sub(r"\[.*?\]", "", ln).strip()
        L += [(int(m) * 60 + float(s), txt) for m, s in stamps]
    return sorted(L)

def av_read(path, rate):
    """Her formatı (m4a, aac, opus, wma...) PyAV/FFmpeg ile float32 stereoya çevirir."""
    import av
    with av.open(path) as c:
        st = c.streams.audio[0]; sr = rate or st.codec_context.sample_rate
        rs = av.AudioResampler(format="flt", layout="stereo", rate=sr); ch = []
        def grab(o):
            for f in (o if isinstance(o, list) else [o]):
                if f is not None: ch.append(f.to_ndarray().reshape(-1, 2))
        for fr in c.decode(st): grab(rs.resample(fr))
        grab(rs.resample(None))
    return np.ascontiguousarray(np.concatenate(ch).astype("float32")), sr

def load_fit(path, hifi):
    if os.path.splitext(path)[1].lower() in SF_EXTS:
        try:
            d, sr = sf.read(path, dtype="float32", always_2d=True)
            d = d[:, :2] if d.shape[1] > 1 else np.repeat(d, 2, 1)
            tgt = sr if hifi else 44100
            if tgt != sr:
                x = np.linspace(0, len(d) - 1, int(len(d) * tgt / sr)); xp = np.arange(len(d))
                d = np.stack([np.interp(x, xp, d[:, k]) for k in (0, 1)], 1).astype("float32")
            return np.ascontiguousarray(d), tgt
        except Exception:
            pass
    return av_read(path, None if hifi else 44100)


class Engine:
    """float32 akış; geçişli yumuşatma (crossfade) ve boşluksuz geçiş."""
    def __init__(s):
        s.data = s.nxt = s.stream = None
        s.sr, s.pos, s.xf, s.vol = 44100, 0, 0, .8
        s.playing = s.swapped = s.ended = False
        s.hifi, s.fmt, s.mixed = True, None, False

    def open(s, sr):
        if s.stream: s.stream.close()
        s.sr, s.fmt = sr, s.hifi
        s.stream = sd.OutputStream(samplerate=sr, channels=2, blocksize=1024, callback=s.cb,
                                   dtype="float32" if s.fmt else "int16")
        s.stream.start()

    def cb(s, out, n, t, st):
        buf = np.zeros((n, 2), np.float32)
        if s.playing and s.data is not None:
            end = len(s.data); a = s.data[s.pos:s.pos + n]; buf[:len(a)] = a
            nxt = s.nxt
            if nxt is not None and s.xf > 0 and end - s.pos < s.xf + n:
                idx = np.arange(s.pos, s.pos + n); j = idx - (end - s.xf)
                f = np.clip(j / s.xf, 0, 1)
                buf *= np.cos(f * np.pi / 2)[:, None]; s.mixed = True
                ok = (j >= 0) & (j < len(nxt))
                buf[ok] += nxt[j[ok]] * np.sin(f[ok] * np.pi / 2)[:, None]
            s.pos += n
            if s.pos >= end:
                if nxt is not None:
                    s.pos = max(s.pos - (end - s.xf), 0) if s.mixed else max(s.pos - end, 0)
                    s.data, s.nxt, s.swapped, s.mixed = nxt, None, True, False
                else:
                    s.playing, s.ended = False, True
        buf *= s.vol
        np.clip(buf, -1, 1, out=buf)
        out[:] = buf if s.fmt else (buf * 32767).astype(np.int16)


# ---------- Dalgalı ilerleme çubuğu (M3 Expressive) ----------
class Wave(Canvas):
    def __init__(s, master, seek):
        super().__init__(master, height=30, highlightthickness=0, bd=0)
        s.frac, s.phase, s.amp, s.tgt, s.seek, s.c = 0, 0, 0, 0, seek, ("#6750A4", "#CAC4D0")
        s.bind("<Button-1>", s.click); s.bind("<B1-Motion>", s.click)

    def click(s, e): s.seek(min(max(e.x / max(s.winfo_width(), 1), 0), 1))

    def draw(s):
        s.delete("all"); w, h = s.winfo_width(), s.winfo_height(); m = h / 2
        s.amp += (s.tgt - s.amp) * .15; s.phase += .18 * (s.amp > .05)
        x = s.frac * w; s.create_line(x, m, w - 3, m, fill=s.c[1], width=5, capstyle="round")
        pts = [v for px in range(3, int(x) + 1, 3) for v in (px, m + s.amp * math.sin(px / 9 - s.phase))]
        if len(pts) >= 4: s.create_line(*pts, fill=s.c[0], width=5, capstyle="round", smooth=True)
        s.create_line(x, 5, x, h - 5, fill=s.c[0], width=6, capstyle="round")


# ---------- Uygulama ----------
class App(ctk.CTk):
    """Kartlar arka plan görselinin içine çizilir (gerçek yuvarlak köşe + gölge).
    Widget'lar kartların içindeki düz renkli 'bölge' çerçevelerine yerleşir."""

    def __init__(s):
        super().__init__()
        s.title("Pixel Müzik"); s.geometry("1080x720"); s.minsize(940, 660)
        try: s.cfg = {**DEF, **json.load(open(CFG))}
        except Exception: s.cfg = dict(DEF)
        s.R, s.W, s.H, s.pr, s.q = s.cfg["radius"], 1080, 720, 36, queue.Queue()
        s.tracks, s.idx, s.pref, s.shuffle, s.repeat = [], 0, None, False, 0
        s.cover, s.seed, s.art_pil, s.bg_pil = gen_cover("x"), SEED, None, None
        s.lrc, s.lt, s.lcur, s.lmsg, s.pt = [], [], -2, "", 0
        s.eng = Engine(); s.eng.hifi = s.cfg["hifi"]; s.eng.vol = s.cfg["vol"]
        s.flat, s.txt, s.sub, s.icons, s.filled, s.sliders, s.switches, s.segs, s.tiles = [], [], [], [], [], [], [], [], []
        s.rows = {}
        s.set_mode(); s.c = palette(SEED, s.dark())
        s.bg = ctk.CTkLabel(s, text=""); s.bg.place(x=0, y=0, relwidth=1, relheight=1)
        s.build_now(); s.build_right(); s.layout()
        s.bind("<Configure>", s.on_cfg)
        s.bind("<space>", lambda e: e.widget.winfo_class() != "Entry" and s.toggle())
        s.protocol("WM_DELETE_WINDOW", s.close)
        s.paint(s.c); s.refresh(); s.tick()
        if s.cfg["folder"]: s.scan(s.cfg["folder"])

    # --- tema / yerleşim ---
    def set_mode(s): ctk.set_appearance_mode({"Açık": "light", "Koyu": "dark", "Sistem": "system"}[s.cfg["theme"]])
    def dark(s): return ctk.get_appearance_mode() == "Dark"
    def post(s, fn): s.q.put(fn)            # iş parçacığından güvenli arayüz çağrısı
    def rects(s): return [(20, 20, 410, s.H - 40), (450, 20, s.W - 470, s.H - 40)]

    def layout(s):
        p = 12 + int(s.R * .3)              # köşe kesimine giren alanı boş bırak
        s.now.configure(width=410 - 2 * p, height=s.H - 40 - 2 * p); s.now.place(x=20 + p, y=20 + p)
        s.right.configure(width=s.W - 470 - 2 * p, height=s.H - 40 - 2 * p); s.right.place(x=450 + p, y=20 + p)

    def paint(s, c):
        s.c = c; card = c["card"]; trk = lerp(c["container"], c["variant"], .3)
        for f in s.flat: f.configure(fg_color=card)
        for t in s.tiles: t.configure(fg_color=lerp(card, c["container"], .6))
        for w in s.txt: w.configure(text_color=c["on_surface"])
        for w in s.sub: w.configure(text_color=c["variant"])
        for b in s.icons:
            lit = b.on or not b.toggle
            b.configure(fg_color=c["container"] if lit else "transparent", hover_color=c["container"],
                        text_color=c["on_container"] if lit else c["variant"])
        for b in s.filled + [s.play_btn]:
            b.configure(fg_color=c["primary"], hover_color=c["primary"], text_color=c["on_primary"])
        for w in s.sliders:
            w.configure(fg_color=trk, progress_color=c["primary"], button_color=c["primary"], button_hover_color=c["primary"])
        for w in s.switches:
            w.configure(fg_color=c["variant"], progress_color=c["primary"], button_color=c["on_primary"],
                        button_hover_color=c["on_primary"], text_color=c["on_surface"])
        for w in s.segs:
            w.configure(fg_color=c["container"], selected_color=c["primary"], selected_hover_color=c["primary"],
                        unselected_color=c["container"], unselected_hover_color=lerp(c["container"], c["primary"], .18),
                        text_color=c["on_container"])
        s.search.configure(fg_color=c["container"], text_color=c["on_container"], border_color=c["container"],
                           placeholder_text_color=c["variant"])
        s.lib.configure(fg_color=card)
        try: s.lib._scrollbar.configure(fg_color=card, button_color=c["container"], button_hover_color=c["variant"])
        except Exception: pass
        s.wave.configure(bg=card); s.wave.c = (c["primary"], c["container"])
        for k, l in enumerate(s.ll):
            l.configure(text_color=c["primary"] if k == 2 else c["variant"] if k in (1, 3) else lerp(c["variant"], card, .55))
        s.segfix()

    def segfix(s):
        """Seçili sekmenin yazısı açık, diğerleri koyu renk olsun (her iki temada okunur)."""
        for g in s.segs:
            cur = g.get()
            for n, b in g._buttons_dict.items():
                b.configure(text_color=s.c["on_primary"] if n == cur else s.c["on_container"])

    def anim(s, ms, step, done=None):
        t0 = time.time()
        def f():
            t = min((time.time() - t0) * 1000 / ms, 1); step(1 - (1 - t) ** 3)   # ease-out
            if t < 1: s.after(30, f)
            elif done: done()
        f()

    def make_bg_img(s, c):
        covers = [t["cover"] for t in s.tracks[:12]] or [s.cover]
        return make_bg(s.cover, covers, s.W, s.H, s.cfg["bg"], c, s.R, s.rects())

    def art_img(s, c):
        """Cosmic Swirl modunda kapak, kütüphane kapaklarından oluşan sarmal kolajın ortasında durur."""
        if s.cfg["bg"] != "Cosmic Swirl": return rounded(s.cover, ART, s.R)
        base = ImageOps.fit(s.cover, (ART, ART)).filter(ImageFilter.GaussianBlur(18))
        base = Image.blend(base, Image.new("RGB", (ART, ART), c["container"]), .5)
        im = swirl(base, [t["cover"] for t in s.tracks[:12]] or [s.cover], max(s.R // 3, 4), .9).convert("RGBA")
        mid = rounded(s.cover, int(ART * .52), s.R // 2); im.paste(mid, ((ART - mid.width) // 2,) * 2, mid)
        return rounded(im.convert("RGB"), ART, s.R)

    def refresh(s):
        """Palet, kapak ve arka plan aynı anda, aynı eğriyle geçiş yapar (kart rengi kaymaz)."""
        new, old = palette(s.seed, s.dark()), dict(s.c)
        art, bg = s.art_img(new), s.make_bg_img(new)
        s.pt += 1; tok = s.pt
        s.anim(380, lambda k: tok == s.pt and s.paint({n: lerp(old[n], new[n], k) for n in new}), s.mark)
        s.fade(s.art, "art_pil", art, (ART, ART)); s.fade(s.bg, "bg_pil", bg, (s.W, s.H))

    def update_bg(s): s.fade(s.bg, "bg_pil", s.make_bg_img(s.c), (s.W, s.H))

    def fade(s, label, attr, new, size):
        old = getattr(s, attr); setattr(s, attr, new)
        tok = label._ftok = getattr(label, "_ftok", 0) + 1
        if old is None or old.size != new.size or old.mode != new.mode:
            label.configure(image=ctk.CTkImage(new, size=size)); return
        s.anim(380, lambda k: tok == label._ftok and label.configure(image=ctk.CTkImage(Image.blend(old, new, k), size=size)))

    def on_cfg(s, e):
        if e.widget is s and (e.width, e.height) != (s.W, s.H) and e.width > 300:
            s.W, s.H = e.width, e.height; s.layout()
            if getattr(s, "_j", None): s.after_cancel(s._j)
            s._j = s.after(150, s.update_bg)

    # --- arayüz ---
    def fr(s, p, **k):
        f = ctk.CTkFrame(p, fg_color=s.c["card"], corner_radius=0, **k); s.flat.append(f); return f

    def lab(s, p, size, bold=False, sub=False, **k):
        l = ctk.CTkLabel(p, text="", font=ctk.CTkFont(size=size, weight="bold" if bold else "normal"), **k)
        (s.sub if sub else s.txt).append(l); return l

    def ib(s, p, text, cmd, size=22, toggle=False):
        b = ctk.CTkButton(p, text=text, width=54, height=54, corner_radius=27, fg_color="transparent",
                          font=ctk.CTkFont(size=size), command=cmd)
        b.toggle, b.on = toggle, False; s.icons.append(b); return b

    def build_now(s):
        f = s.now = s.fr(s, width=10, height=10); f.pack_propagate(False)
        s.art = ctk.CTkLabel(f, text="", width=ART, height=ART); s.art.pack(pady=(8, 16))
        s.title_l = s.lab(f, 24, True); s.title_l.pack()
        s.artist_l = s.lab(f, 15, sub=True); s.artist_l.pack(pady=(2, 12))
        s.wave = Wave(f, s.seek); s.wave.pack(fill="x", padx=6, pady=(4, 0))
        r = s.fr(f); r.pack(fill="x", padx=8)
        s.t1, s.t2 = s.lab(r, 12, sub=True), s.lab(r, 12, sub=True)
        s.t1.configure(text="0:00"); s.t2.configure(text="0:00"); s.t1.pack(side="left"); s.t2.pack(side="right")
        row = s.fr(f); row.pack(pady=(16, 10))
        s.b_shuf = s.ib(row, "⇄", s.on_shuffle, 20, True); s.b_shuf.pack(side="left")
        s.ib(row, "⏮", s.prev).pack(side="left")
        s.play_btn = ctk.CTkButton(row, text="▶", width=104, height=72, corner_radius=36, font=ctk.CTkFont(size=26), command=s.toggle)
        s.play_btn.pack(side="left", padx=6)
        s.ib(row, "⏭", s.next).pack(side="left")
        s.b_rep = s.ib(row, "↻", s.on_repeat, 20, True); s.b_rep.pack(side="left")
        v = ctk.CTkSlider(f, from_=0, to=1, width=200, command=s.set_vol); v.set(s.cfg["vol"]); v.pack(pady=(10, 0)); s.sliders.append(v)

    def build_right(s):
        f = s.right = s.fr(s, width=10, height=10); f.pack_propagate(False)
        seg = ctk.CTkSegmentedButton(f, values=["Kitaplık", "Sözler", "Ayarlar"], command=lambda v: (s.tab(v), s.segfix()), height=38); s.segs.append(seg)
        seg.pack(pady=(4, 12), fill="x"); seg.set("Kitaplık")
        s.pages = {n: s.fr(f) for n in ("Kitaplık", "Sözler", "Ayarlar")}
        p = s.pages["Kitaplık"]; top = s.fr(p); top.pack(fill="x")
        s.search = ctk.CTkEntry(top, placeholder_text="Şarkı veya sanatçı ara", height=40, corner_radius=20, border_width=0)
        s.search.pack(side="left", fill="x", expand=True); s.search.bind("<KeyRelease>", lambda e: s.render_list())
        b = ctk.CTkButton(top, text="Klasör seç", width=100, height=40, corner_radius=20, command=s.pick)
        b.pack(side="left", padx=(8, 0)); s.filled.append(b)
        s.lib = ctk.CTkScrollableFrame(p, fg_color=s.c["card"]); s.lib.pack(fill="both", expand=True, pady=(8, 0))
        s.ll = [s.lab(s.pages["Sözler"], 22 if k == 2 else 16, k == 2, sub=True, wraplength=440) for k in range(5)]
        for l in s.ll: l.pack(expand=True, pady=6)
        st = s.pages["Ayarlar"]
        def row(text, w):
            r = ctk.CTkFrame(st, corner_radius=18, height=56); r.pack(fill="x", pady=5); r.pack_propagate(False); s.tiles.append(r)
            s.lab(r, 14).configure(text=text); s.txt[-1].pack(side="left", padx=(18, 0)); w(r).pack(side="right", padx=16)
        def seg_(vals, key, cb):
            def mk(r):
                g = ctk.CTkSegmentedButton(r, values=vals, command=lambda v: (cb(v), s.segfix())); g.set(s.cfg[key]); s.segs.append(g); return g
            return mk
        def sw(key, cb=None):
            def mk(r):
                v = ctk.CTkSwitch(r, text="", width=46, command=lambda: (s.cfg.__setitem__(key, bool(v.get())), cb and cb()))
                v.select() if s.cfg[key] else v.deselect(); s.switches.append(v); return v
            return mk
        def sl(a, b, key, cb, n):
            def mk(r):
                v = ctk.CTkSlider(r, from_=a, to=b, number_of_steps=n, width=200, command=cb); v.set(s.cfg[key]); s.sliders.append(v); return v
            return mk
        row("Tema", seg_(["Açık", "Koyu", "Sistem"], "theme", lambda v: (s.cfg.__setitem__("theme", v), s.set_mode(), s.refresh())))
        row("Albüm kapağından renk", sw("dyn", lambda: (setattr(s, "seed", dominant(s.cover) if s.cfg["dyn"] else SEED), s.refresh())))
        row("Arka plan", seg_(["Bulanık", "Cosmic Swirl"], "bg", lambda v: (s.cfg.__setitem__("bg", v), s.refresh())))
        row("Köşe yarıçapı", sl(0, 40, "radius", s.set_radius, 20))
        row("Geçişli yumuşatma (sn)", sl(0, 12, "xf", s.set_xf, 12))
        row("Hi-Fi modu (float32)", sw("hifi", s.set_hifi))
        row("Çevrimiçi söz indir", sw("online"))
        s.tab("Kitaplık")

    def tab(s, name):
        for p in s.pages.values(): p.pack_forget()
        s.pages[name].pack(fill="both", expand=True)

    def set_radius(s, v):
        s.R = int(v); s.cfg["radius"] = s.R; s.layout()
        if getattr(s, "_rj", None): s.after_cancel(s._rj)
        s._rj = s.after(180, s.refresh)

    def set_vol(s, x): s.eng.vol = x; s.cfg["vol"] = x
    def set_xf(s, v): s.cfg["xf"] = int(v); s.eng.xf = int(v * s.eng.sr)

    def set_hifi(s):
        s.eng.hifi = s.cfg["hifi"]
        if s.eng.data is not None: s.play(s.idx, s.eng.pos / len(s.eng.data))

    # --- kitaplık ---
    def pick(s):
        d = filedialog.askdirectory()
        if d: s.cfg["folder"] = d; s.scan(d)

    def scan(s, folder):
        def work():
            fs = sorted(os.path.join(r, f) for r, _, fl in os.walk(folder) for f in fl if f.lower().endswith(EXTS))
            tr = [meta(p) for p in fs[:500]]
            s.post(lambda: (setattr(s, "tracks", tr), s.render_list(), s.refresh()))
        threading.Thread(target=work, daemon=True).start()

    def render_list(s):
        for w in s.lib.winfo_children(): w.destroy()
        s.rows = {}; q, n = s.search.get().lower(), 0
        if not s.tracks:
            ctk.CTkLabel(s.lib, text="Müziklerini eklemek için\n“Klasör seç”e dokun.", text_color=s.c["variant"],
                         font=ctk.CTkFont(size=14)).pack(pady=60); return
        for i, t in enumerate(s.tracks):
            if q and q not in (t["title"] + t["artist"]).lower(): continue
            th = t.setdefault("thumb", rounded(t["cover"], 52, 14))
            b = ctk.CTkButton(s.lib, text=f'{t["title"]}\n{t["artist"]}', anchor="w", height=64, corner_radius=min(s.R, 22),
                              image=ctk.CTkImage(th, size=(52, 52)), compound="left", font=ctk.CTkFont(size=13),
                              fg_color="transparent", text_color=s.c["on_surface"], hover_color=s.c["container"],
                              command=lambda i=i: s.play(i))
            b.pack(fill="x", pady=2); s.rows[i] = b
            try: b._text_label.configure(justify="left")
            except Exception: pass
            n += 1
            if n >= 150: break
        s.mark()

    def mark(s):
        for i, b in s.rows.items():
            cur = i == s.idx
            b.configure(fg_color=s.c["container"] if cur else "transparent", text_color=s.c["on_container"] if cur else s.c["on_surface"])

    # --- çalma ---
    def next_index(s, auto=True):
        n = len(s.tracks)
        if not n: return None
        if s.repeat == 2 and auto: return s.idx
        if s.shuffle: return random.randrange(n)
        return s.idx + 1 if s.idx + 1 < n else (0 if s.repeat else None)

    def play(s, i, frac=0.0):
        s.idx = i; t = s.tracks[i]; s.show(t)
        def work():
            try: d, sr = load_fit(t["path"], s.cfg["hifi"])
            except Exception:
                s.post(lambda: s.artist_l.configure(text="Dosya açılamadı (pip install av kurulu mu?)")); return
            s.post(lambda: s.begin(i, d, sr, frac))
        threading.Thread(target=work, daemon=True).start()

    def begin(s, i, d, sr, frac):
        if i != s.idx: return
        e = s.eng; e.playing = False
        try:
            if e.stream is None or sr != e.sr or e.fmt != e.hifi: e.open(sr)
        except Exception:
            s.artist_l.configure(text="Ses aygıtı açılamadı"); return
        e.data, e.nxt, e.pos, e.mixed, e.xf = d, None, int(frac * len(d)), False, int(s.cfg["xf"] * sr)
        e.playing = True; s.play_state(); s.prefetch()

    def prefetch(s):
        j = s.pref = s.next_index()
        if j is None: return
        cur = s.idx
        def work():
            try: d, sr = load_fit(s.tracks[j]["path"], s.cfg["hifi"])
            except Exception: return
            if s.idx == cur and s.pref == j and sr == s.eng.sr: s.eng.nxt = d
        threading.Thread(target=work, daemon=True).start()

    def show(s, t):
        cut = lambda x: x if len(x) <= 26 else x[:25] + "…"
        s.title_l.configure(text=cut(t["title"])); s.artist_l.configure(text=cut(t["artist"]))
        s.cover = t["cover"]; s.seed = dominant(s.cover) if s.cfg["dyn"] else SEED
        s.refresh(); s.load_lyrics(t)

    def toggle(s):
        e = s.eng
        if e.data is None:
            if s.tracks: s.play(0)
            return
        e.playing = not e.playing; s.play_state()

    def play_state(s):
        playing = s.eng.playing; tgt = 22 if playing else 36; a, s.pr = s.pr, tgt   # şekil dönüşümü
        s.play_btn.configure(text="⏸" if playing else "▶")
        s.anim(280, lambda k: s.play_btn.configure(corner_radius=int(a + (tgt - a) * k)))
        s.wave.tgt = 4 if playing else 0

    def next(s):
        j = s.next_index(False)
        if j is not None: s.play(j)

    def prev(s):
        e = s.eng
        if e.data is not None and (e.pos / e.sr > 3 or s.idx == 0): e.pos = 0
        elif s.idx > 0: s.play(s.idx - 1)

    def seek(s, f):
        if s.eng.data is not None: s.eng.pos = int(f * len(s.eng.data))

    def on_shuffle(s): s.shuffle = not s.shuffle; s.b_shuf.on = s.shuffle; s.paint(s.c)
    def on_repeat(s):
        s.repeat = (s.repeat + 1) % 3
        s.b_rep.configure(text="↻¹" if s.repeat == 2 else "↻"); s.b_rep.on = bool(s.repeat); s.paint(s.c)

    # --- sözler ---
    def load_lyrics(s, t):
        s.lrc, s.lt, s.lcur, s.lmsg = [], [], -2, "Sözler aranıyor…"; s.lyr_render()
        def work():
            p = os.path.splitext(t["path"])[0] + ".lrc"; txt = None
            try:
                if os.path.exists(p): txt = open(p, encoding="utf-8", errors="ignore").read()
                elif s.cfg["online"] and syncedlyrics:
                    txt = syncedlyrics.search(f'{t["artist"]} {t["title"]}', synced_only=True)
                    if txt: open(p, "w", encoding="utf-8").write(txt)
            except Exception: pass
            L = parse_lrc(txt) if txt else []
            def done():
                if s.tracks and s.tracks[s.idx] is t:
                    s.lrc, s.lt, s.lcur = L, [x[0] for x in L], -2
                    if not L: s.lmsg = "Bu şarkı için söz bulunamadı"; s.lyr_render()
            s.post(done)
        threading.Thread(target=work, daemon=True).start()

    def lyr_render(s):
        i = s.lcur
        for k, l in enumerate(s.ll):
            j = i + k - 2
            l.configure(text=s.lrc[j][1] if 0 <= j < len(s.lrc) else (s.lmsg if k == 2 and not s.lrc else ""))
        a, b = s.c["variant"], s.c["primary"]; tok = s._lt = getattr(s, "_lt", 0) + 1
        s.anim(300, lambda k: tok == s._lt and s.ll[2].configure(text_color=lerp(a, b, k)))

    # --- döngü ---
    def tick(s):
        try:
            try:
                while True: s.q.get_nowait()()
            except queue.Empty: pass
            e = s.eng
            if e.swapped:
                e.swapped = False; s.idx = s.pref if s.pref is not None else s.idx
                s.show(s.tracks[s.idx]); s.prefetch()
            if e.ended:
                e.ended = False; s.play_state()
                if s.pref is not None: s.play(s.pref)
            if e.data is not None:
                pos, dur = e.pos / e.sr, len(e.data) / e.sr
                s.wave.frac = min(pos / max(dur, 1), 1)
                s.t1.configure(text="%d:%02d" % divmod(int(pos), 60)); s.t2.configure(text="%d:%02d" % divmod(int(dur), 60))
                if s.lt:
                    i = bisect.bisect_right(s.lt, pos + .15) - 1
                    if i != s.lcur: s.lcur = i; s.lyr_render()
            s.wave.draw()
        finally:
            s.after(33, s.tick)

    def close(s):
        try: json.dump(s.cfg, open(CFG, "w"))
        except Exception: pass
        if s.eng.stream: s.eng.stream.close()
        s.destroy()


if __name__ == "__main__":
    App().mainloop()
