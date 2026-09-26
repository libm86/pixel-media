"""PixelPlayer - Modern, Animasyonlu ve Gelişmiş Material You Müzik Çalar"""
import bisect, colorsys, io, json, math, os, queue, random, re, threading, time
import numpy as np
import customtkinter as ctk
import sounddevice as sd
import soundfile as sf
from tkinter import Canvas, filedialog
from mutagen import File as MFile
from PIL import Image, ImageDraw, ImageFilter, ImageOps

print("PixelMedia for PC : 2.0.0")

try:
    import syncedlyrics
except ImportError:
    print("SyncedLyrics is not avaible. ImportError")
    syncedlyrics = None

EXTS = (".mp3", ".flac", ".wav", ".ogg", ".opus", ".m4a", ".aac", ".wma", ".aiff", ".aif", ".alac", ".mka", ".webm", ".mp4")
SF_EXTS = (".wav", ".flac", ".ogg", ".mp3", ".aiff", ".aif")
CFG = os.path.join(os.path.expanduser("~"), ".pixel_player_cfg.json")
DEF = dict(folder="", theme="Koyu", dyn=True, bg="Bulanık", radius=24, xf=4, hifi=True, online=True, vol=0.8, favorites=[])
SEED = (103, 80, 164)
ART = 360

# ---------- Renk ve Görsel Yardımcılar ----------
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
    s = min(max(s, 0.40), 0.85)
    t = (dict(primary=.80, on_primary=.20, container=.25, on_container=.90, surface=.05, card=.10, on_surface=.95, variant=.70)
         if dark else
         dict(primary=.40, on_primary=.99, container=.90, on_container=.15, surface=.95, card=.90, on_surface=.10, variant=.40))
    return {k: hexc(h, s * (.30 if k in ("surface", "card") else 1), v) for k, v in t.items()}

def rounded(img, size, r):
    img = ImageOps.fit(img, (size, size), Image.LANCZOS).convert("RGBA")
    m = Image.new("L", (size * 4, size * 4), 0)
    ImageDraw.Draw(m).rounded_rectangle((0, 0, size * 4 - 1, size * 4 - 1), min(r, size // 2) * 4, fill=255)
    img.putalpha(m.resize((size, size), Image.LANCZOS))
    return img

def gen_cover(seed):
    rnd = random.Random(seed); h = rnd.random()
    x = np.linspace(0, 1, 256); g = ((x[None, :] + x[:, None]) / 2)[..., None]
    c1 = np.array(colorsys.hls_to_rgb(h, .55, .8)) * 255
    c2 = np.array(colorsys.hls_to_rgb((h + .25) % 1, .72, .8)) * 255
    return Image.fromarray((c1 * (1 - g) + c2 * g).astype("uint8"))

def make_bg(cover, w, h, c):
    s = 4; w2, h2 = max(w // s, 8), max(h // s, 8)
    im = ImageOps.fit(cover, (w2, h2)).filter(ImageFilter.GaussianBlur(35))
    tint = Image.new("RGB", (w2, h2), c["surface"])
    im = Image.blend(im, tint, .75)
    return im.resize((w, h), Image.BICUBIC)

# ---------- Ses ve Meta İşlemleri ----------
def meta(path):
    t = dict(path=path, title=os.path.splitext(os.path.basename(path))[0], artist="Bilinmeyen Sanatçı", cover=None)
    try:
        m = MFile(path); e = MFile(path, easy=True)
        if e:
            t["title"] = e.get("title", [t["title"]])[0]; t["artist"] = e.get("artist", [t["artist"]])[0]
        raw = m.pictures[0].data if getattr(m, "pictures", None) else None
        if raw is None and m.tags:
            raw = next((m.tags[k].data for k in m.tags.keys() if k.startswith("APIC")), None)
        if raw is None and m.tags and "covr" in m.tags: raw = bytes(m.tags["covr"][0])
        if raw:
            t["cover"] = Image.open(io.BytesIO(raw)).convert("RGB"); t["cover"].thumbnail((500, 500))
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
    import av
    with av.open(path) as c:
        st = c.streams.audio[0]; sr = st.codec_context.sample_rate if hifi else 44100
        rs = av.AudioResampler(format="flt", layout="stereo", rate=sr); ch = []
        def grab(o):
            for f in (o if isinstance(o, list) else [o]):
                if f is not None: ch.append(f.to_ndarray().reshape(-1, 2))
        for fr in c.decode(st): grab(rs.resample(fr))
        grab(rs.resample(None))
    return np.ascontiguousarray(np.concatenate(ch).astype("float32")), sr

class Engine:
    def __init__(s):
        s.data = s.nxt = s.stream = None
        s.sr, s.pos, s.xf, s.vol = 44100, 0, 0, .8
        s.playing = s.swapped = s.ended = False
        s.hifi, s.fmt, s.mixed = True, None, False
        s.muted = False

    def open(s, sr):
        if s.stream: s.stream.close()
        s.sr, s.fmt = sr, s.hifi
        s.stream = sd.OutputStream(samplerate=sr, channels=2, blocksize=1024, callback=s.cb, dtype="float32" if s.fmt else "int16")
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
        
        current_vol = 0.0 if s.muted else s.vol
        buf *= current_vol
        np.clip(buf, -1, 1, out=buf)
        out[:] = buf if s.fmt else (buf * 32767).astype(np.int16)

# ---------- Geliştirilmiş İlerleme Dalga Çubuğu ----------
class Wave(Canvas):
    def __init__(s, master, seek):
        super().__init__(master, height=40, highlightthickness=0, bd=0)
        s.frac, s.phase, s.amp, s.tgt, s.seek, s.c = 0, 0, 0, 0, seek, ("#6750A4", "#CAC4D0")
        s.bind("<Button-1>", s.click); s.bind("<B1-Motion>", s.click)

    def click(s, e): s.seek(min(max(e.x / max(s.winfo_width(), 1), 0), 1))

    def draw(s):
        s.delete("all"); w, h = s.winfo_width(), s.winfo_height(); m = h / 2
        s.amp += (s.tgt - s.amp) * .15; s.phase += .20 * (s.amp > .05)
        x = s.frac * w
        
        # Arka plan barı (oynatılmayan kısım)
        s.create_line(x, m, w, m, fill=s.c[1], width=4, capstyle="round")
        
        # Animasyonlu arka dalga
        pts_bg = [v for px in range(2, int(x) + 1, 4) for v in (px, m + (s.amp * 0.5) * math.sin(px / 15 - s.phase * 1.5))]
        if len(pts_bg) >= 4: s.create_line(*pts_bg, fill=s.c[1], width=3, capstyle="round", smooth=True)
        
        # Ana dalga
        pts = [v for px in range(2, int(x) + 1, 3) for v in (px, m + s.amp * math.sin(px / 10 - s.phase))]
        if len(pts) >= 4: s.create_line(*pts, fill=s.c[0], width=5, capstyle="round", smooth=True)
        
        # Playhead (İlerleme noktası)
        s.create_oval(x - 6, m - 6, x + 6, m + 6, fill=s.c[0], outline="#ffffff", width=2)

# ---------- Ana Uygulama ----------
class App(ctk.CTk):
    def __init__(s):
        super().__init__()
        s.title("PixelPlayer"); s.geometry("1180x780"); s.minsize(980, 680)
        try: s.cfg = {**DEF, **json.load(open(CFG))}
        except Exception: s.cfg = dict(DEF)
        
        s.W, s.H, s.q = 1180, 780, queue.Queue()
        s.tracks, s.idx, s.pref, s.shuffle, s.repeat = [], 0, None, False, 0
        s.cover, s.seed, s.bg_pil = gen_cover("x"), SEED, None
        s.lrc, s.lt, s.lcur, s.lmsg, s.pt = [], [], -2, "", 0
        s.eng = Engine(); s.eng.hifi = s.cfg["hifi"]; s.eng.vol = s.cfg["vol"]
        
        # UI Element Listeleri (Tema geçişleri için)
        s.texts, s.subtexts, s.cards, s.primary_btns, s.icons, s.segs = [], [], [], [], [], []
        s.list_rows = {}
        
        s.set_mode(); s.c = palette(SEED, s.dark())
        
        # Arka plan
        s.bg = ctk.CTkLabel(s, text=""); s.bg.place(x=0, y=0, relwidth=1, relheight=1)
        
        # Ana Layout: Top (Sidebar + Content) ve Bottom (Player)
        s.main_area = ctk.CTkFrame(s, fg_color="transparent")
        s.main_area.pack(side="top", fill="both", expand=True)
        
        s.bottom_bar = ctk.CTkFrame(s, height=100, corner_radius=0)
        s.bottom_bar.pack(side="bottom", fill="x")
        s.cards.append(s.bottom_bar)
        
        s.build_sidebar()
        s.build_content()
        s.build_player()
        
        s.nav_to("Library")
        
        s.bind("<Configure>", s.on_cfg)
        s.bind("<space>", lambda e: e.widget.winfo_class() != "Entry" and s.toggle())
        s.protocol("WM_DELETE_WINDOW", s.close)
        
        s.paint(s.c); s.refresh(); s.tick()
        if s.cfg["folder"]: s.scan(s.cfg["folder"])

    # --- Sistem ve Tema ---
    def set_mode(s): ctk.set_appearance_mode({"Açık": "light", "Koyu": "dark", "Sistem": "system"}[s.cfg["theme"]])
    def dark(s): return ctk.get_appearance_mode() == "Dark"
    def post(s, fn): s.q.put(fn)

    def paint(s, c):
        s.c = c; card = c["card"]
        for f in s.cards: f.configure(fg_color=card)
        for w in s.texts: w.configure(text_color=c["on_surface"])
        for w in s.subtexts: w.configure(text_color=c["variant"])
        for b in s.primary_btns: b.configure(fg_color=c["primary"], hover_color=lerp(c["primary"], "#ffffff", 0.15), text_color=c["on_primary"])
        for b in s.icons:
            lit = getattr(b, "on", False)
            b.configure(fg_color=c["container"] if lit else "transparent", hover_color=c["container"], text_color=c["on_container"] if lit else c["variant"])
        
        s.search.configure(fg_color=c["container"], text_color=c["on_container"], border_color=c["container"], placeholder_text_color=c["variant"])
        s.vol_slider.configure(fg_color=lerp(c["container"], c["variant"], .3), progress_color=c["primary"], button_color=c["primary"])
        s.wave.configure(bg=card); s.wave.c = (c["primary"], c["container"])
        
        s.sidebar.configure(fg_color=lerp(c["surface"], c["card"], 0.5))
        s.lib.configure(fg_color="transparent")
        
        # Aktif menü sekmesi rengi
        for name, btn in s.nav_btns.items():
            active = getattr(btn, "active", False)
            btn.configure(fg_color=c["container"] if active else "transparent", text_color=c["on_container"] if active else c["on_surface"])

        s.update_vol_icon()
        s.mark()

    def anim(s, ms, step, done=None):
        t0 = time.time()
        def f():
            t = min((time.time() - t0) * 1000 / ms, 1); step(1 - (1 - t) ** 3)
            if t < 1: s.after(16, f)
            elif done: done()
        f()

    def refresh(s):
        new, old = palette(s.seed, s.dark()), dict(s.c)
        bg = make_bg(s.cover, s.W, s.H, new)
        s.pt += 1; tok = s.pt
        s.anim(400, lambda k: tok == s.pt and s.paint({n: lerp(old[n], new[n], k) for n in new}))
        
        old_bg = getattr(s, "bg_pil", None); s.bg_pil = bg
        if old_bg is None or old_bg.size != bg.size:
            s.bg.configure(image=ctk.CTkImage(bg, size=(s.W, s.H))); return
        
        btok = s.bg._ftok = getattr(s.bg, "_ftok", 0) + 1
        s.anim(400, lambda k: btok == s.bg._ftok and s.bg.configure(image=ctk.CTkImage(Image.blend(old_bg, bg, k), size=(s.W, s.H))))
        
        # Kapak resmi animasyonu
        art = rounded(s.cover, ART, s.cfg["radius"])
        s.art_img.configure(image=ctk.CTkImage(art, size=(ART, ART)))
        
        small_art = rounded(s.cover, 64, 12)
        s.p_cover.configure(image=ctk.CTkImage(small_art, size=(64, 64)))

    def on_cfg(s, e):
        if e.widget is s and (e.width, e.height) != (s.W, s.H) and e.width > 300:
            s.W, s.H = e.width, e.height
            if getattr(s, "_j", None): s.after_cancel(s._j)
            s._j = s.after(250, s.refresh)

    # --- UI İnşası ---
    def build_sidebar(s):
        s.sidebar = ctk.CTkFrame(s.main_area, width=220, corner_radius=0)
        s.sidebar.pack(side="left", fill="y")
        s.sidebar.pack_propagate(False)
        
        logo = ctk.CTkLabel(s.sidebar, text="🎵 PixelPlayer", font=ctk.CTkFont(size=22, weight="bold"))
        logo.pack(pady=(30, 40), padx=20, anchor="w")
        s.texts.append(logo)
        
        s.nav_btns = {}
        def nav_btn(text, icon, page):
            b = ctk.CTkButton(s.sidebar, text=f"  {icon}  {text}", anchor="w", height=45, corner_radius=12,
                              font=ctk.CTkFont(size=15, weight="bold"), hover_color=s.c["container"], command=lambda: s.nav_to(page))
            b.pack(fill="x", padx=15, pady=5)
            s.nav_btns[page] = b
        
        nav_btn("Şu An Çalan", "🎧", "NowPlaying")
        nav_btn("Kitaplık", "📂", "Library")
        nav_btn("Sözler", "🎤", "Lyrics")
        nav_btn("Ayarlar", "⚙️", "Settings")
        
        s.nav_btns["Library"].active = True

    def nav_to(s, page):
        for p, b in s.nav_btns.items():
            b.active = (p == page)
        s.paint(s.c)
        for p in s.pages.values(): p.pack_forget()
        s.pages[page].pack(fill="both", expand=True)

    def build_content(s):
        s.content = ctk.CTkFrame(s.main_area, fg_color="transparent")
        s.content.pack(side="right", fill="both", expand=True)
        s.pages = {n: ctk.CTkFrame(s.content, fg_color="transparent") for n in ("NowPlaying", "Library", "Lyrics", "Settings")}
        
        # 1. Şu An Çalan (Now Playing)
        np_frame = s.pages["NowPlaying"]
        s.art_img = ctk.CTkLabel(np_frame, text="")
        s.art_img.pack(expand=True, pady=(20, 0))
        
        # 2. Kitaplık (Library)
        lib_page = s.pages["Library"]
        top = ctk.CTkFrame(lib_page, fg_color="transparent", height=60)
        top.pack(fill="x", padx=30, pady=(30, 10))
        
        title = ctk.CTkLabel(top, text="Tüm Şarkılar", font=ctk.CTkFont(size=28, weight="bold"))
        title.pack(side="left")
        s.texts.append(title)
        
        b = ctk.CTkButton(top, text="Klasör Seç", width=120, height=40, corner_radius=20, command=s.pick)
        b.pack(side="right", padx=(10, 0)); s.primary_btns.append(b)
        
        s.search = ctk.CTkEntry(top, placeholder_text="Şarkı veya sanatçı ara...", width=250, height=40, corner_radius=20, border_width=0)
        s.search.pack(side="right"); s.search.bind("<KeyRelease>", lambda e: s.render_list())
        
        s.lib = ctk.CTkScrollableFrame(lib_page)
        s.lib.pack(fill="both", expand=True, padx=20, pady=10)
        
        # 3. Sözler (Lyrics)
        lyr_page = s.pages["Lyrics"]
        s.ll = [ctk.CTkLabel(lyr_page, font=ctk.CTkFont(size=28 if k == 2 else 18, weight="bold" if k == 2 else "normal"), wraplength=700) for k in range(7)]
        for l in s.ll:
            l.pack(expand=True, pady=10)
            if l != s.ll[3]: s.subtexts.append(l) # Orta sözler hariç hepsi subtext renginde
            
        # 4. Ayarlar (Settings)
        st = s.pages["Settings"]
        ctk.CTkLabel(st, text="Uygulama Ayarları", font=ctk.CTkFont(size=28, weight="bold")).pack(anchor="w", padx=40, pady=(30, 20))
        
        def row(text, w):
            r = ctk.CTkFrame(st, corner_radius=18, height=64); r.pack(fill="x", padx=40, pady=8); r.pack_propagate(False); s.cards.append(r)
            l = ctk.CTkLabel(r, text=text, font=ctk.CTkFont(size=15)); l.pack(side="left", padx=25)
            s.texts.append(l); w(r).pack(side="right", padx=20)
            
        def sw(key, cb=None):
            def mk(r):
                v = ctk.CTkSwitch(r, text="", command=lambda: (s.cfg.__setitem__(key, bool(v.get())), cb and cb()))
                v.select() if s.cfg[key] else v.deselect(); return v
            return mk
            
        def seg_(vals, key, cb):
            def mk(r):
                g = ctk.CTkSegmentedButton(r, values=vals, command=cb); g.set(s.cfg[key]); s.segs.append(g); return g
            return mk

        row("Tema", seg_(["Açık", "Koyu", "Sistem"], "theme", lambda v: (s.cfg.__setitem__("theme", v), s.set_mode(), s.refresh())))
        row("Albüm kapağından renk uyarlaması", sw("dyn", lambda: (setattr(s, "seed", dominant(s.cover) if s.cfg["dyn"] else SEED), s.refresh())))
        row("Hi-Fi Ses Modu (Gelişmiş)", sw("hifi", s.set_hifi))
        row("Çevrimiçi Sözleri İndir", sw("online"))

    def build_player(s):
        s.p_cover = ctk.CTkLabel(s.bottom_bar, text="")
        s.p_cover.pack(side="left", padx=(20, 15), pady=18)
        
        info_frame = ctk.CTkFrame(s.bottom_bar, fg_color="transparent", width=200)
        info_frame.pack(side="left", fill="y", pady=25)
        
        s.p_title = ctk.CTkLabel(info_frame, text="Şarkı Seçilmedi", font=ctk.CTkFont(size=16, weight="bold"), anchor="w")
        s.p_title.pack(fill="x")
        s.texts.append(s.p_title)
        
        s.p_artist = ctk.CTkLabel(info_frame, text="-", font=ctk.CTkFont(size=13), anchor="w")
        s.p_artist.pack(fill="x")
        s.subtexts.append(s.p_artist)
        
        # Favori Butonu
        s.fav_btn = ctk.CTkButton(s.bottom_bar, text="♡", width=30, height=30, fg_color="transparent", font=ctk.CTkFont(size=22), command=s.toggle_favorite)
        s.fav_btn.pack(side="left", padx=10)
        s.texts.append(s.fav_btn)
        
        # Orta Kısım (Kontroller ve Dalga)
        mid_frame = ctk.CTkFrame(s.bottom_bar, fg_color="transparent")
        mid_frame.pack(side="left", expand=True, fill="both", padx=20)
        
        ctrl_frame = ctk.CTkFrame(mid_frame, fg_color="transparent")
        ctrl_frame.pack(pady=(12, 0))
        
        def ic(p, txt, cmd, size=20, tog=False):
            b = ctk.CTkButton(p, text=txt, width=40, height=40, corner_radius=20, fg_color="transparent", font=ctk.CTkFont(size=size), command=cmd)
            b.toggle, b.on = tog, False; s.icons.append(b); return b
            
        s.b_shuf = ic(ctrl_frame, "⇄", s.on_shuffle, 18, True); s.b_shuf.pack(side="left", padx=10)
        ic(ctrl_frame, "⏮", s.prev, 18).pack(side="left", padx=5)
        
        s.play_btn = ctk.CTkButton(ctrl_frame, text="▶", width=56, height=56, corner_radius=28, font=ctk.CTkFont(size=24), command=s.toggle)
        s.play_btn.pack(side="left", padx=15); s.primary_btns.append(s.play_btn)
        
        ic(ctrl_frame, "⏭", s.next, 18).pack(side="left", padx=5)
        s.b_rep = ic(ctrl_frame, "↻", s.on_repeat, 18, True); s.b_rep.pack(side="left", padx=10)
        
        wave_frame = ctk.CTkFrame(mid_frame, fg_color="transparent")
        wave_frame.pack(fill="x", pady=(0, 5))
        
        s.t1 = ctk.CTkLabel(wave_frame, text="0:00", font=ctk.CTkFont(size=11), width=35); s.t1.pack(side="left")
        s.subtexts.append(s.t1)
        
        s.wave = Wave(wave_frame, s.seek)
        s.wave.pack(side="left", expand=True, fill="x", padx=10)
        
        s.t2 = ctk.CTkLabel(wave_frame, text="0:00", font=ctk.CTkFont(size=11), width=35); s.t2.pack(side="right")
        s.subtexts.append(s.t2)
        
        # Sağ Kısım (Ses)
        right_frame = ctk.CTkFrame(s.bottom_bar, fg_color="transparent", width=200)
        right_frame.pack(side="right", fill="y", padx=20, pady=35)
        
        s.vol_icon = ctk.CTkButton(right_frame, text="🔊", width=30, height=30, fg_color="transparent", font=ctk.CTkFont(size=18), command=s.toggle_mute)
        s.vol_icon.pack(side="left"); s.icons.append(s.vol_icon)
        
        s.vol_slider = ctk.CTkSlider(right_frame, from_=0, to=1, width=120, command=s.set_vol)
        s.vol_slider.set(s.cfg["vol"]); s.vol_slider.pack(side="left", padx=5)

    # --- Ses ve Kontroller ---
    def set_vol(s, x): 
        s.eng.vol = x; s.cfg["vol"] = x; s.eng.muted = False
        s.update_vol_icon()

    def toggle_mute(s):
        s.eng.muted = not s.eng.muted
        s.update_vol_icon()

    def update_vol_icon(s):
        if s.eng.muted or s.eng.vol == 0: icon = "🔇"
        elif s.eng.vol < 0.4: icon = "🔉"
        else: icon = "🔊"
        s.vol_icon.configure(text=icon, text_color=s.c["variant"] if s.eng.muted else s.c["on_surface"])

    def set_hifi(s):
        s.eng.hifi = s.cfg["hifi"]
        if s.eng.data is not None: s.play(s.idx, s.eng.pos / len(s.eng.data))

    # --- Kitaplık ve Dosyalar ---
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
        s.list_rows = {}; q, n = s.search.get().lower(), 0
        if not s.tracks:
            ctk.CTkLabel(s.lib, text="Müzik klasörü boş veya seçilmedi.", text_color=s.c["variant"], font=ctk.CTkFont(size=16)).pack(pady=100); return
            
        for i, t in enumerate(s.tracks):
            if q and q not in (t["title"] + t["artist"]).lower(): continue
            th = t.setdefault("thumb", rounded(t["cover"], 48, 12))
            
            row = ctk.CTkFrame(s.lib, height=70, corner_radius=12, fg_color="transparent")
            row.pack(fill="x", pady=3, padx=10)
            row.pack_propagate(False)
            
            # Hover Eventleri
            def on_e(e, r=row): r.configure(fg_color=s.c["container"])
            def on_l(e, r=row, idx=i): r.configure(fg_color=s.c["container"] if idx == s.idx else "transparent")
            
            row.bind("<Enter>", on_e); row.bind("<Leave>", on_l)
            row.bind("<Button-1>", lambda e, idx=i: s.play(idx))
            
            img = ctk.CTkLabel(row, text="", image=ctk.CTkImage(th, size=(48, 48)))
            img.pack(side="left", padx=10); img.bind("<Button-1>", lambda e, idx=i: s.play(idx))
            
            t_frame = ctk.CTkFrame(row, fg_color="transparent")
            t_frame.pack(side="left", fill="both", expand=True, padx=10)
            t_frame.bind("<Button-1>", lambda e, idx=i: s.play(idx))
            
            title_l = ctk.CTkLabel(t_frame, text=t["title"], font=ctk.CTkFont(size=15, weight="bold"), anchor="w")
            title_l.pack(fill="x", pady=(12, 0)); title_l.bind("<Button-1>", lambda e, idx=i: s.play(idx))
            
            art_l = ctk.CTkLabel(t_frame, text=t["artist"], font=ctk.CTkFont(size=12), text_color=s.c["variant"], anchor="w")
            art_l.pack(fill="x"); art_l.bind("<Button-1>", lambda e, idx=i: s.play(idx))
            
            play_ico = ctk.CTkLabel(row, text="🎵" if i == s.idx else "", font=ctk.CTkFont(size=20), text_color=s.c["primary"])
            play_ico.pack(side="right", padx=20)
            
            s.list_rows[i] = (row, title_l, play_ico)
            n += 1
            if n >= 200: break
        s.mark()

    def mark(s):
        for i, (row, title_l, play_ico) in s.list_rows.items():
            cur = i == s.idx
            row.configure(fg_color=s.c["container"] if cur else "transparent")
            title_l.configure(text_color=s.c["on_container"] if cur else s.c["on_surface"])
            play_ico.configure(text="🎵" if cur else "")

    def toggle_favorite(s):
        if not s.tracks: return
        p = s.tracks[s.idx]["path"]
        if p in s.cfg.get("favorites", []):
            s.cfg["favorites"].remove(p)
            s.fav_btn.configure(text="♡", text_color=s.c["on_surface"])
        else:
            if "favorites" not in s.cfg: s.cfg["favorites"] = []
            s.cfg["favorites"].append(p)
            s.fav_btn.configure(text="♥", text_color="#FF4B4B")
        json.dump(s.cfg, open(CFG, "w"))

    # --- Oynatma ---
    def next_index(s, auto=True):
        n = len(s.tracks)
        if not n: return None
        if s.repeat == 2 and auto: return s.idx
        if s.shuffle: return random.randrange(n)
        return s.idx + 1 if s.idx + 1 < n else (0 if s.repeat else None)

    def play(s, i, frac=0.0):
        s.idx = i; t = s.tracks[i]; s.show(t); s.mark()
        
        # Favori ikonunu güncelle
        fav = "♥" if t["path"] in s.cfg.get("favorites", []) else "♡"
        c_fav = "#FF4B4B" if fav == "♥" else s.c["on_surface"]
        s.fav_btn.configure(text=fav, text_color=c_fav)

        def work():
            try: d, sr = load_fit(t["path"], s.cfg["hifi"])
            except Exception: s.post(lambda: s.p_title.configure(text="Dosya açılamadı!")); return
            s.post(lambda: s.begin(i, d, sr, frac))
        threading.Thread(target=work, daemon=True).start()

    def begin(s, i, d, sr, frac):
        if i != s.idx: return
        e = s.eng; e.playing = False
        try:
            if e.stream is None or sr != e.sr or e.fmt != e.hifi: e.open(sr)
        except Exception: return
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
        cut = lambda x: x if len(x) <= 30 else x[:28] + "…"
        s.p_title.configure(text=cut(t["title"])); s.p_artist.configure(text=cut(t["artist"]))
        s.cover = t["cover"]; s.seed = dominant(s.cover) if s.cfg["dyn"] else SEED
        s.refresh(); s.load_lyrics(t)

    def toggle(s):
        e = s.eng
        if e.data is None:
            if s.tracks: s.play(0)
            return
        e.playing = not e.playing; s.play_state()

    def play_state(s):
        playing = s.eng.playing
        s.play_btn.configure(text="⏸" if playing else "▶")
        s.wave.tgt = 6 if playing else 0

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

    # --- Şarkı Sözleri (Lyrics) ---
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
                    if not L: s.lmsg = "Bu şarkı için söz bulunamadı 🎶"; s.lyr_render()
            s.post(done)
        threading.Thread(target=work, daemon=True).start()

    def lyr_render(s):
        i = s.lcur
        for k, l in enumerate(s.ll):
            j = i + k - 3 # Merkez = index 3
            l.configure(text=s.lrc[j][1] if 0 <= j < len(s.lrc) else (s.lmsg if k == 3 and not s.lrc else ""))
            
            # Dinamik büyüklük ve renk (Ortadaki söz vurgulanır)
            if k == 3:
                l.configure(text_color=s.c["primary"], font=ctk.CTkFont(size=32, weight="bold"))
            else:
                l.configure(text_color=lerp(s.c["variant"], s.c["surface"], abs(3-k) * 0.2), font=ctk.CTkFont(size=18))

    def tick(s):
        try:
            try:
                while True: s.q.get_nowait()()
            except queue.Empty: pass
            
            e = s.eng
            if e.swapped:
                e.swapped = False; s.idx = s.pref if s.pref is not None else s.idx
                s.show(s.tracks[s.idx]); s.prefetch(); s.mark()
            if e.ended:
                e.ended = False; s.play_state()
                if s.pref is not None: s.play(s.pref)
                
            if e.data is not None:
                pos, dur = e.pos / e.sr, len(e.data) / e.sr
                s.wave.frac = min(pos / max(dur, 1), 1)
                s.t1.configure(text="%d:%02d" % divmod(int(pos), 60)); s.t2.configure(text="%d:%02d" % divmod(int(dur), 60))
                
                if s.lt:
                    i = bisect.bisect_right(s.lt, pos + .2) - 1
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
    print("Program loaded into RAM, opening...")
    App().mainloop()
