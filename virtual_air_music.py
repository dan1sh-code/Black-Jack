#!/usr/bin/env python3
"""
🎵 VIRTUAL AIR MUSIC - ERGONOMIC & MIRROR-FIXED VERSION 🎵
Optimized for comfort, accurate left/right mapping, and sweep-to-play interaction.
Requirements: pip install opencv-python mediapipe pygame numpy
Controls: q=Quit | m=Menu | p=Practice | 1-6=Songs | r=Reset
"""

import cv2
import mediapipe as mp
import pygame
import numpy as np
import time

# ======================== CONFIG ========================
SCREEN_W, SCREEN_H = 1280, 720
COLORS = {"bg": (15,15,25), "cyan": (0,255,255), "pink": (255,0,255), 
          "green": (0,255,100), "orange": (255,165,0), "white": (255,255,255)}

GENRES = {
    "LATIHAN": {"name": "LATIHAN DoReMi", "color": (0,255,100), "bpm": 90,
        "left": {"KICK":(80,0.25,"drum_kick"), "SNARE":(250,0.12,"drum_snare"), 
                 "HIHAT":(6000,0.06,"drum_hihat"), "CLAP":(1500,0.08,"drum_clap")},
        "right": {"DO":(261.63,0.4,"sine"), "RE":(293.66,0.4,"sine"), "MI":(329.63,0.4,"sine"),
                  "FA":(349.23,0.4,"sine"), "SOL":(392.0,0.4,"sine"), "LA":(440.0,0.4,"sine"),
                  "SI":(493.88,0.4,"sine"), "DO'":(523.25,0.4,"sine")}},
    "POP": {"name": "POP Mode", "color": (255,200,50), "bpm": 120,
        "left": {"KICK":(60,0.3,"drum_kick"), "SNARE":(200,0.15,"drum_snare"), 
                 "HIHAT":(7000,0.05,"drum_hihat"), "CLAP":(1800,0.1,"drum_clap")},
        "right": {"C4":(261.63,0.35,"bright"), "E4":(329.63,0.35,"bright"), "G4":(392.0,0.35,"bright"),
                  "A4":(440.0,0.35,"bright"), "C5":(523.25,0.35,"bright"), "D5":(587.33,0.35,"bright"),
                  "E5":(659.25,0.35,"bright"), "G5":(783.99,0.35,"bright")}},
    "ROCK": {"name": "ROCK Mode", "color": (0,50,255), "bpm": 140,
        "left": {"KICK":(50,0.3,"drum_kick_heavy"), "SNARE":(180,0.2,"drum_snare_heavy"), 
                 "HIHAT":(5000,0.08,"drum_hihat"), "CRASH":(3000,0.5,"drum_crash")},
        "right": {"E2":(82.41,0.5,"distortion"), "A2":(110.0,0.5,"distortion"), "D3":(146.83,0.5,"distortion"),
                  "G3":(196.0,0.5,"distortion"), "B3":(246.94,0.5,"distortion"), "E4":(329.63,0.5,"distortion"),
                  "PWR1":(110.0,0.6,"power_chord"), "PWR2":(146.83,0.6,"power_chord")}},
    "DJ": {"name": "DJ Mode", "color": (255,0,255), "bpm": 128,
        "left": {"KICK":(45,0.35,"drum_808"), "SNARE":(200,0.15,"drum_trap"), 
                 "HIHAT":(9000,0.04,"drum_hihat"), "SUB":(30,0.5,"sub_bass")},
        "right": {"SYN1":(261.63,0.3,"synth_saw"), "SYN2":(329.63,0.3,"synth_saw"), "SYN3":(392.0,0.3,"synth_saw"),
                  "SYN4":(523.25,0.3,"synth_saw"), "PAD1":(220.0,0.6,"synth_pad"), "PAD2":(293.66,0.6,"synth_pad"),
                  "FX1":(1000,0.5,"fx_riser"), "FX2":(2000,0.3,"fx_laser")}},
    "JAZZ": {"name": "JAZZ Mode", "color": (200,150,50), "bpm": 100,
        "left": {"KICK":(70,0.25,"drum_kick"), "SNARE":(220,0.15,"drum_brush"), 
                 "HIHAT":(6500,0.06,"drum_hihat"), "RIDE":(4000,0.3,"drum_ride")},
        "right": {"Cmaj7":(261.63,0.5,"jazz_chord"), "Dm7":(293.66,0.5,"jazz_chord"), "Em7":(329.63,0.5,"jazz_chord"),
                  "Fmaj7":(349.23,0.5,"jazz_chord"), "G7":(392.0,0.5,"jazz_chord"), "Am7":(440.0,0.5,"jazz_chord"),
                  "Bdim":(493.88,0.5,"jazz_chord"), "C5":(523.25,0.5,"jazz_chord")}},
    "CLASSICAL": {"name": "CLASSICAL Mode", "color": (180,130,255), "bpm": 80,
        "left": {"TIMPANI":(65,0.4,"drum_timpani"), "CYMBAL":(3500,0.5,"drum_crash"), 
                 "BASS_D":(73.42,0.4,"drum_timpani"), "TRIANGLE":(5000,0.3,"triangle_hit")},
        "right": {"C4":(261.63,0.6,"piano"), "D4":(293.66,0.6,"piano"), "E4":(329.63,0.6,"piano"),
                  "F4":(349.23,0.6,"piano"), "G4":(392.0,0.6,"piano"), "A4":(440.0,0.6,"piano"),
                  "B4":(493.88,0.6,"piano"), "C5":(523.25,0.6,"piano")}}
}

PRACTICE_SONGS = {
    "Twinkle Twinkle": {"notes": ["DO","DO","SOL","SOL","LA","LA","SOL","-","FA","FA","MI","MI","RE","RE","DO","-"], "timing": 0.5},
    "Do Re Mi Basic": {"notes": ["DO","RE","MI","DO","DO","RE","MI","DO","MI","FA","SOL","-","MI","FA","SOL","-"], "timing": 0.4},
    "Happy Birthday": {"notes": ["DO","DO","RE","DO","FA","MI","-","DO","DO","RE","DO","SOL","FA","-"], "timing": 0.5},
    "Basic Beat 4/4": {"notes": ["KICK","HIHAT","SNARE","HIHAT","KICK","HIHAT","SNARE","HIHAT"], "timing": 0.3},
    "Rock Beat": {"notes": ["KICK","HIHAT","SNARE","HIHAT","KICK","KICK","SNARE","CRASH"], "timing": 0.25},
    "DJ Drop": {"notes": ["KICK","KICK","KICK","KICK","SNARE","-","KICK","SUB"], "timing": 0.25}
}

# ======================== SOUND GENERATOR ========================
class SoundGen:
    def __init__(self, sr=44100):
        self.sr = sr
        self.cache = {}
    def _t(self, d): return np.linspace(0, d, int(self.sr*d), False)
    def _env(self, t, decay): return np.exp(-t*decay)
    def _make(self, wave, vol=0.7):
        if np.max(np.abs(wave)) > 0: wave = wave / np.max(np.abs(wave))
        wave = wave * vol
        fade = min(int(0.02*self.sr), len(wave)//4)
        wave[-fade:] *= np.linspace(1, 0, fade)
        audio = np.zeros((len(wave), 2), dtype=np.int16)
        audio[:,0] = audio[:,1] = (wave * 32767).astype(np.int16)
        return pygame.sndarray.make_sound(audio)
    def gen(self, freq, dur, stype):
        key = f"{freq}_{dur}_{stype}"
        if key in self.cache: return self.cache[key]
        t = self._t(dur)
        w = np.sin(2*np.pi*freq*t)
        if stype == "bright": w = w*0.6 + np.sin(2*np.pi*freq*2*t)*0.25 + np.sin(2*np.pi*freq*3*t)*0.1; w *= self._env(t, 3)
        elif stype == "distortion": 
            for i in range(2, 8): w += np.sin(2*np.pi*freq*i*t)*(0.5/i)
            w = np.clip(w*3, -1, 1) * self._env(t, 2)
        elif stype == "power_chord": w = np.sin(2*np.pi*freq*t)*0.5 + np.sin(2*np.pi*freq*1.5*t)*0.3 + np.sin(2*np.pi*freq*2*t)*0.2; w = np.clip(w*4, -1, 1) * self._env(t, 1.5)
        elif stype == "synth_saw": 
            for i in range(1, 12): w += np.sin(2*np.pi*freq*i*t)*((-1)**(i+1))/i
            w *= 0.5 * self._env(t, 4)
        elif stype == "synth_pad": w = (np.sin(2*np.pi*freq*t)*0.4 + np.sin(2*np.pi*freq*1.005*t)*0.3 + np.sin(2*np.pi*freq*0.995*t)*0.3); w *= (1 - np.exp(-t*5)) * self._env(t, 1)
        elif stype == "piano": w = (w*0.5 + np.sin(2*np.pi*freq*2*t)*0.25 + np.sin(2*np.pi*freq*3*t)*0.125) * self._env(t, 4)
        elif stype == "jazz_chord": w = w*0.3 + np.sin(2*np.pi*freq*1.26*t)*0.2 + np.sin(2*np.pi*freq*1.5*t)*0.2 + np.sin(2*np.pi*freq*1.78*t)*0.15; w *= self._env(t, 2.5)
        elif "drum_kick" in stype:
            fs = freq * np.exp(-t * (30 if "heavy" not in stype else 20))
            w = np.sin(2*np.pi*fs*t) * self._env(t, 12 if "heavy" not in stype else 8)
            w += np.random.randn(len(t)) * np.exp(-t*200) * 0.3
            if "heavy" in stype: w = np.clip(w*2, -1, 1)
        elif stype == "drum_808": fs=freq*np.exp(-t*15); w=np.sin(2*np.pi*fs*t)*self._env(t,5)+np.sin(2*np.pi*30*t)*self._env(t,3)*0.5
        elif "snare" in stype: w = np.sin(2*np.pi*freq*t)*self._env(t,25) + np.random.randn(len(t))*self._env(t, 15 if "heavy" not in stype else 10)*0.6
        elif stype == "drum_hihat": w = (np.random.randn(len(t))*0.7 + np.sin(2*np.pi*freq*t)*0.3) * self._env(t, 40) * 0.5
        elif stype == "drum_clap": n=np.random.randn(len(t)); w=n*(self._env(t,50)*0.5+np.exp(-(t-0.02)**2*5000)*0.3+np.exp(-(t-0.04)**2*3000)*0.8)*0.6
        elif stype == "drum_crash": w = (np.random.randn(len(t))*0.6 + np.sin(2*np.pi*freq*t)*0.3) * self._env(t, 4) * 0.5
        elif stype == "drum_ride": w = (np.sin(2*np.pi*freq*t)*0.4 + np.sin(2*np.pi*freq*2.3*t)*0.4 + np.random.randn(len(t))*0.2) * self._env(t, 5) * 0.5
        elif stype == "drum_timpani": w = (np.sin(2*np.pi*freq*t) + np.sin(2*np.pi*freq*1.5*t)*0.3) * self._env(t, 3) * 0.7
        elif stype == "triangle_hit": w = np.sin(2*np.pi*freq*t) * self._env(t, 6) * 0.4
        elif stype == "sub_bass": w = np.sin(2*np.pi*freq*t) * self._env(t, 2) * 0.9
        elif stype == "fx_riser": fs=freq*(1+t*5); ph=2*np.pi*np.cumsum(fs)/self.sr; w=np.sin(ph)*(t/dur)*0.6
        elif stype == "fx_laser": fs=freq*np.exp(-t*10); ph=2*np.pi*np.cumsum(fs)/self.sr; w=np.sin(ph)*self._env(t,5)*0.5
        elif stype == "drum_brush": n=np.random.randn(len(t)); w=np.convolve(n,np.ones(20)/20,mode='same')*self._env(t,8)*0.5
        else: w *= self._env(t, 4)
        snd = self._make(w)
        self.cache[key] = snd
        return snd

# ======================== VISUAL EFFECTS ========================
class Particle:
    def __init__(self, x, y, c, spd=3, life=0.8):
        self.x, self.y, self.c = x, y, c
        self.vx, self.vy = np.random.uniform(-spd, spd), np.random.uniform(-spd, spd)
        self.life, self.max_life, self.birth, self.sz = life, life, time.time(), np.random.randint(2, 8)
    def update(self):
        self.x += self.vx; self.y += self.vy; self.vy += 0.1
        self.life = self.max_life - (time.time() - self.birth)
        return self.life > 0
    def draw(self, img):
        if self.life <= 0: return
        a = max(0, self.life / self.max_life); s = max(1, int(self.sz * a))
        cv2.circle(img, (int(self.x), int(self.y)), s, tuple(int(v*a) for v in self.c), -1)

class VFX:
    def __init__(self, w, h):
        self.w, self.h = w, h
        self.particles, self.flashes, self.ripples, self.hits = [], [], [], []
        self.spectrum = np.zeros(64); self.beat = 0; self.frm = 0
    def spawn(self, x, y, c, n=15, s=4): 
        for _ in range(n): self.particles.append(Particle(x, y, c, s))
    def flash(self, c, i=0.2, d=0.12): 
        self.flashes.append({"c":c, "i":i, "t":time.time(), "d":d})
    def ripple(self, x, y, c, r=80): 
        self.ripples.append({"x":x, "y":y, "c":c, "r":10, "mr":r, "t":time.time(), "d":0.4})
    def hit(self, n, c): 
        self.hits.append({"n":n, "c":c, "t":time.time()})
        if len(self.hits) > 20: self.hits.pop(0)
        self.beat = min(1.0, self.beat + 0.25)
    def update(self):
        self.frm += 1
        self.particles = [p for p in self.particles if p.update()]
        self.beat *= 0.93
        self.spectrum = self.spectrum * 0.85
        if self.beat > 0.1: self.spectrum += np.random.rand(64) * self.beat * 25
    def draw(self, img):
        now = time.time()
        bw = max(1, (self.w-400)//64); bx, by = 200, self.h-40
        for i, v in enumerate(self.spectrum):
            h = max(1, int(v))
            cv2.rectangle(img, (bx+i*bw, by-h), (bx+i*bw+bw-1, by), 
                         (min(255,int(100+h*2)), max(0,min(255,255-int(h*2))), min(255,int(h*3))), -1)
        for p in self.particles: p.draw(img)
        self.flashes = [f for f in self.flashes if now-f["t"] < f["d"]]
        for f in self.flashes:
            a = f["i"] * (1 - (now-f["t"])/f["d"])
            ov = img.copy(); ov[:] = f["c"]; cv2.addWeighted(ov, a, img, 1-a, 0, img)
        self.ripples = [r for r in self.ripples if now-r["t"] < r["d"]]
        for r in self.ripples:
            p = (now-r["t"]) / r["d"]; rad = int(r["r"] + (r["mr"]-r["r"])*p); a = 1 - p
            cv2.circle(img, (r["x"], r["y"]), rad, tuple(int(v*a) for v in r["c"]), max(1, int(3*a)))
        cv2.putText(img, "TIMELINE:", (250, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150,150,150), 1)
        for i, h in enumerate([h for h in self.hits if now-h["t"]<5][-12:]):
            x = 250 + i*70; a = max(0.2, 1-(now-h["t"])/5); c = tuple(int(v*a) for v in h["c"])
            cv2.rectangle(img, (x,70), (x+60,95), c, -1); cv2.rectangle(img, (x,70), (x+60,95), (255,255,255), 1)
            cv2.putText(img, h["n"][:5], (x+3,88), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255), 1)
        if self.beat > 0.05:
            cv2.rectangle(img, (0,0), (self.w-1,self.h-1), (int(self.beat*200),0,int(self.beat*200)), max(1,int(self.beat*6)))

class UI:
    @staticmethod
    def box(img, txt, x, y, w, h, c, act=False, hov=False):
        ov = img.copy()
        if act:
            cv2.rectangle(ov, (x,y), (x+w,y+h), c, -1); cv2.addWeighted(ov, 0.6, img, 0.4, 0, img)
            cv2.rectangle(img, (x-4,y-4), (x+w+4,y+h+4), c, 3)
        elif hov:
            cv2.rectangle(ov, (x,y), (x+w,y+h), c, -1); cv2.addWeighted(ov, 0.25, img, 0.75, 0, img)
            cv2.rectangle(img, (x,y), (x+w,y+h), c, 2)
        else:
            cv2.rectangle(ov, (x,y), (x+w,y+h), c, -1); cv2.addWeighted(ov, 0.08, img, 0.92, 0, img)
            cv2.rectangle(img, (x,y), (x+w,y+h), c, 2)
        fs = 0.65 if len(txt) > 6 else 0.85
        ts = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, fs, 2)[0]
        cv2.putText(img, txt, (x+(w-ts[0])//2, y+(h+ts[1])//2), cv2.FONT_HERSHEY_SIMPLEX, fs, (255,255,255), 2)
    @staticmethod
    def title(img, txt, y=40, c=(0,255,255), s=1.0):
        ts = cv2.getTextSize(txt, cv2.FONT_HERSHEY_DUPLEX, s, 2)[0]; x = (img.shape[1] - ts[0]) // 2
        cv2.putText(img, txt, (x+2,y+2), cv2.FONT_HERSHEY_DUPLEX, s, (0,0,0), 3)
        cv2.putText(img, txt, (x,y), cv2.FONT_HERSHEY_DUPLEX, s, c, 2)
    @staticmethod
    def cursor(img, x, y, c=(0,255,255)):
        cv2.circle(img, (x,y), 20, c, 2); cv2.circle(img, (x,y), 12, c, -1)
        cv2.circle(img, (x,y), 28, tuple(v//2 for v in c), 1)
    @staticmethod
    def practice(img, name, notes, idx, y=650):
        cv2.putText(img, f"Practice: {name}", (350, y-25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)
        for i, n in enumerate(notes):
            x = 350 + i * 55
            if x > 1200: break
            c = (0,255,0) if i < idx else (255,255,0) if i == idx else (100,100,100)
            cv2.rectangle(img, (x,y), (x+50,y+30), c, 2 if i==idx else 1)
            cv2.putText(img, n[:4], (x+3, y+22), cv2.FONT_HERSHEY_SIMPLEX, 0.4, c, 1)

# ======================== GAME STATE ========================
class Game:
    def __init__(self):
        self.state = "menu"; self.genre = None; self.sounds = {}; self.cooldowns = {}; self.cd = 0.18
        self.boxes_l = {}; self.boxes_r = {}; self.hover = {}; self.hover_delay = 0.8
        self.prac_active = False; self.prac_song = None; self.prac_idx = 0; self.prac_score = 0
        self.hits = 0; self.start = time.time()
    def load(self, g):
        self.genre = g; self.sounds = {}; self.cooldowns = {}; self.state = "playing"
        data = GENRES[g]
        for n, (f, d, t) in {**data["left"], **data["right"]}.items(): self.sounds[n] = sg.gen(f, d, t)
        self._calc_boxes(data)
    def _calc_boxes(self, data):
        self.boxes_l = {}; self.boxes_r = {}
        # ERGONOMIC LAYOUT: Posisi lebih rendah & dekat tengah (seperti drum/keyboard)
        lh = list(data["left"].items()); bh = 90
        for i, (n, d) in enumerate(lh):
            self.boxes_l[n] = {"x": 140, "y": 280 + i*(bh+18), "w": 210, "h": bh, "c": COLORS["cyan"], "k": n, "triggered": False}
        rh = list(data["right"].items()); bhr = 55
        for i, (n, d) in enumerate(rh):
            self.boxes_r[n] = {"x": SCREEN_W-350, "y": 260 + i*(bhr+12), "w": 210, "h": bhr, "c": COLORS["pink"], "k": n, "triggered": False}
    def play(self, n):
        now = time.time()
        if now - self.cooldowns.get(n, 0) > self.cd and n in self.sounds:
            self.sounds[n].play(); self.cooldowns[n] = now; self.hits += 1
            c = self.boxes_l[n]["c"] if n in self.boxes_l else self.boxes_r.get(n, {}).get("c", (0,255,255))
            vfx.hit(n, c); return True
        return False
    def start_prac(self, name):
        if name in PRACTICE_SONGS:
            self.prac_active = True; self.prac_song = name; self.prac_idx = 0; self.prac_score = 0
    def check_prac(self, n):
        if not self.prac_active or not self.prac_song: return
        s = PRACTICE_SONGS[self.prac_song]
        if self.prac_idx < len(s["notes"]) and n == s["notes"][self.prac_idx]:
            self.prac_score += 1; self.prac_idx += 1
            if self.prac_idx >= len(s["notes"]): self.prac_active = False

# ======================== HELPERS ========================
sg = SoundGen(); vfx = VFX(SCREEN_W, SCREEN_H); ui = UI(); game = Game()

def collide(fx, fy, b): return b["x"] < fx < b["x"]+b["w"] and b["y"] < fy < b["y"]+b["h"]

def draw_menu(img):
    ui.title(img, "VIRTUAL AIR MUSIC", 60, (0,255,255), 1.3)
    cv2.putText(img, "Pilih Genre - Arahkan & Tahan Jari 0.8 Detik", (360, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180,180,180), 1)
    genres = list(GENRES.keys()); cols = 3; bw, bh, p = 320, 130, 30
    sx = (SCREEN_W - (cols*bw + (cols-1)*p)) // 2; sy = 140; boxes = {}
    for i, g in enumerate(genres):
        r, c = i//cols, i%cols; x, y = sx + c*(bw+p), sy + r*(bh+p)
        boxes[g] = {"x":x, "y":y, "w":bw, "h":bh}
        hov = g in game.hover; prog = min(1.0, (time.time()-game.hover[g])/game.hover_delay) if hov else 0
        ui.box(img, "", x, y, bw, bh, GENRES[g]["color"], prog>0.9, hov)
        cv2.putText(img, f'{GENRES[g]["name"]}', (x+15, y+40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        if prog > 0: cv2.rectangle(img, (x+10, y+bh-20), (x+10+int((bw-20)*prog), y+bh-10), GENRES[g]["color"], -1)
    cv2.putText(img, "Practice Songs (tekan 1-6 saat bermain):", (sx, sy+2*(bh+p)+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,200,0), 1)
    for i, s in enumerate(list(PRACTICE_SONGS.keys())[:6]): cv2.putText(img, f"{i+1}. {s}", (sx+i*170, sy+2*(bh+p)+45), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200,200,200), 1)
    return boxes

def draw_play(img):
    g = GENRES[game.genre]; ui.title(img, f'{g["name"]}', 35, g["color"], 0.8)
    # Ergonomic Labels
    cv2.putText(img, "👈 KIRI (DRUM/BEAT)", (140, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,200,255), 1)
    cv2.putText(img, "KANAN (MELODY/SYNTH) 👉", (SCREEN_W-380, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,100,200), 1)
    # Rest Zone
    cv2.rectangle(img, (480, 250), (800, 580), (60,60,70), 2)
    cv2.putText(img, "ZONA ISTIRAHAT (Aman)", (530, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120,120,130), 1)
    for n, b in game.boxes_l.items(): ui.box(img, n, b["x"], b["y"], b["w"], b["h"], b["c"], time.time()-game.cooldowns.get(n,0)<0.12)
    for n, b in game.boxes_r.items(): ui.box(img, n, b["x"], b["y"], b["w"], b["h"], b["c"], time.time()-game.cooldowns.get(n,0)<0.12)
    ui.box(img, "MENU [m]", SCREEN_W//2-60, SCREEN_H-50, 120, 40, (100,100,100))
    cv2.putText(img, f"Hits: {game.hits} | Time: {int(time.time()-game.start)}s | BPM: {g['bpm']}", (SCREEN_W//2-150, SCREEN_H-60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150,150,150), 1)
    if game.prac_active and game.prac_song:
        s = PRACTICE_SONGS[game.prac_song]; ui.practice(img, game.prac_song, s["notes"], game.prac_idx, SCREEN_H-100)
        cv2.putText(img, f"Score: {game.prac_score}/{len(s['notes'])}", (350, SCREEN_H-110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

# ======================== MAIN LOOP ========================
if __name__ == "__main__":
    print("="*50 + "\n🎵 VIRTUAL AIR MUSIC - ERGONOMIC VERSION 🎵\n" + "="*50)
    print("Controls: q=Quit | m=Menu | p=Practice | 1-6=Songs | r=Reset")
    print("Tip: Geser tangan masuk ke zona untuk bunyi. Keluar zona untuk reset.")
    print("     Zona tengah aman untuk istirahat. Lengan bisa santai!")
    
    pygame.mixer.init(44100, -16, 2, 512); pygame.mixer.set_num_channels(32)
    mp_hands = mp.solutions.hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.6)
    cap = cv2.VideoCapture(0); cap.set(3, SCREEN_W); cap.set(4, SCREEN_H)
    prac_keys = list(PRACTICE_SONGS.keys())

    while True:
        ok, img = cap.read()
        if not ok: break
        img = cv2.flip(img, 1)
        img = cv2.addWeighted(img, 0.65, np.zeros_like(img), 0.35, 0)
        res = mp_hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        vfx.update()

        if game.state == "menu":
            mboxes = draw_menu(img); hovers = set()
            if res.multi_hand_landmarks:
                for i, hl in enumerate(res.multi_hand_landmarks):
                    mp.solutions.drawing_utils.draw_landmarks(img, hl, mp.solutions.hands.HAND_CONNECTIONS,
                        mp.solutions.drawing_utils.DrawingSpec(color=(0,200,255), thickness=2, circle_radius=3),
                        mp.solutions.drawing_utils.DrawingSpec(color=(0,100,200), thickness=2))
                    tip = hl.landmark[8]; fx, fy = int(tip.x * SCREEN_W), int(tip.y * SCREEN_H)
                    ui.cursor(img, fx, fy)
                    for g, b in mboxes.items():
                        if collide(fx, fy, b):
                            hovers.add(g)
                            if g not in game.hover: game.hover[g] = time.time()
                            if time.time() - game.hover[g] >= game.hover_delay:
                                game.load(g); game.hover = {}; vfx.flash(GENRES[g]["color"], 0.3, 0.2); break
            game.hover = {k: v for k, v in game.hover.items() if k in hovers}

        elif game.state == "playing":
            draw_play(img)
            if res.multi_hand_landmarks:
                for i, hl in enumerate(res.multi_hand_landmarks):
                    # FIX MIRROR: Pakai posisi X langsung. <50% = Kiri, >50% = Kanan
                    tip = hl.landmark[8]; fx_norm = tip.x
                    side = "Left" if fx_norm < 0.5 else "Right"
                    
                    mp.solutions.drawing_utils.draw_landmarks(img, hl, mp.solutions.hands.HAND_CONNECTIONS,
                        mp.solutions.drawing_utils.DrawingSpec(color=(0,200,255) if side=="Left" else (255,100,200), thickness=2, circle_radius=3),
                        mp.solutions.drawing_utils.DrawingSpec(color=(0,100,200) if side=="Left" else (200,50,150), thickness=2))
                    
                    fx, fy = int(fx_norm * SCREEN_W), int(tip.y * SCREEN_H)
                    cc = (0,200,255) if side=="Left" else (255,100,200)
                    ui.cursor(img, fx, fy, cc)
                    
                    boxes = game.boxes_l if side=="Left" else game.boxes_r
                    for n, b in boxes.items():
                        is_inside = collide(fx, fy, b)
                        # SWEEP-TO-PLAY: Bunyi 1x saat masuk, reset saat keluar. Anti-capek & anti-spam.
                        if is_inside and not b["triggered"]:
                            if game.play(n):
                                b["triggered"] = True
                                vfx.spawn(b["x"]+b["w"]//2, b["y"]+b["h"]//2, b["c"], 12, 3)
                                vfx.ripple(b["x"]+b["w"]//2, b["y"]+b["h"]//2, b["c"], 70)
                                vfx.flash(b["c"], 0.12, 0.08)
                                game.check_prac(n)
                        elif not is_inside:
                            b["triggered"] = False

        vfx.draw(img)
        cv2.putText(img, f"FPS: {int(cap.get(cv2.CAP_PROP_FPS))}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100,100,100), 1)
        cv2.imshow("Virtual Air Music", img)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'): break
        elif k == ord('m'): game.state = "menu"; game.prac_active = False; game.hover = {}
        elif k == ord('r'): game.hits = 0; game.start = time.time(); game.prac_score = 0; game.prac_idx = 0
        elif k == ord('p') and game.state == "playing":
            game.prac_active = not game.prac_active
            if game.prac_active:
                for s in PRACTICE_SONGS: game.start_prac(s); break
        elif ord('1') <= k <= ord('6'):
            idx = k - ord('1')
            if idx < len(prac_keys) and game.state == "playing": game.start_prac(prac_keys[idx])

    cap.release(); cv2.destroyAllWindows(); pygame.quit()
    print("\n✅ Terima kasih sudah bermain Virtual Air Music! 🎵")