#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spectral Text Printer — génère un signal IQ (complexe) dont la waterfall affiche un texte.

Principe:
- Le texte est rendu en image binaire (Pillow).
- Pour chaque ligne de l'image (axe Y), on produit un court segment de signal.
- Les pixels "allumés" sur la ligne activent des sinusoïdes à des fréquences
  réparties uniformément dans [f_min, f_min + bandwidth].
- On concatène les segments pour former un signal IQ complet.

Sorties:
- Un fichier .npy (complex64) contenant le signal IQ.
- Optionnellement un WAV stéréo (I=canal gauche, Q=canal droit) si --wav est fourni.

Dépendances: numpy, pillow (PIL). (scipy n'est pas requis)
"""
import argparse
import numpy as np
from PIL import Image, ImageFont, ImageDraw

def reverse_time(iq: np.ndarray) -> np.ndarray:
    """
    Inverse le signal dans le temps (time-reversal).
    """
    return iq[::-1].copy()

def render_text_bitmap(text, font_size=80, font_path=None, padding=10, invert=False, target_width=None, target_height=None):
    """
    Rend le texte dans une image binaire (True = pixel "allumé").
    - target_width / height permettent de redimensionner (conserver ratio).
    """
    # Crée une image provisoire très large pour dessiner le texte proprement
    tmp_w, tmp_h = 2000, 600
    img = Image.new("L", (tmp_w, tmp_h), color=255)  # blanc
    draw = ImageDraw.Draw(img)

    # Police
    try:
        font = ImageFont.truetype(font_path if font_path else ImageFont.load_default().path, font_size)
    except Exception:
        font = ImageFont.load_default()

    # Taille du texte
    bbox = draw.textbbox((0,0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]

    # Nouvelle image juste à la bonne taille + padding
    W = tw + 2*padding
    H = th + 2*padding
    img = Image.new("L", (W, H), color=255)
    draw = ImageDraw.Draw(img)
    draw.text((padding, padding), text, fill=0, font=font)  # noir sur blanc

    # Redimensionnement si demandé
    if target_width or target_height:
        # conserve le ratio
        ratio = W / H
        if target_width and target_height:
            new_w, new_h = target_width, target_height
        elif target_width:
            new_w = target_width
            new_h = max(1, int(target_width / ratio))
        else:
            new_h = target_height
            new_w = max(1, int(target_height * ratio))
        img = img.resize((new_w, new_h), resample=Image.BILINEAR)

    # Binarisation
    arr = np.array(img)
    # pixels "encre" = arr<128
    mask = (arr < 128)
    if invert:
        mask = ~mask
    return mask  # bool [H, W], True = pixel "allumé"

def synthesize_iq_from_bitmap(bitmap, fs, duration_s, f_min, bandwidth_hz, amp=0.8, per_row_window=True, normalize_per_row=True, dc_guard=True):
    """
    Génère un signal IQ complexe à partir d'une image binaire (H x W).
    - fs: fréquence d'échantillonnage (Hz)
    - duration_s: durée totale (s)
    - f_min: fréquence basse (Hz)
    - bandwidth_hz: bande totale (Hz)
    - amp: amplitude nominale (limiter si beaucoup de tons)
    - per_row_window: applique une petite fenêtre sur chaque ligne pour éviter les clicks
    - normalize_per_row: adapte l'amplitude si beaucoup de pixels allumés
    - dc_guard: évite le tout premier bin (proche DC) pour une meilleure lisibilité
    """
    H, W = bitmap.shape
    samples_total = int(round(fs * duration_s))
    samples_per_row = samples_total // H
    # Ajuste exactement la durée en recollant la dernière ligne
    last_row_extra = samples_total - samples_per_row * (H - 1)

    # Pré-calcule les fréquences correspondant aux colonnes
    # On place la fréquence au centre du "bin" de colonne: (x+0.5)/W
    x_idx = np.arange(W)
    freqs = f_min + (x_idx + 0.5) * (bandwidth_hz / W)

    if dc_guard:
        # Écarte légèrement de DC si f_min≈0
        eps = bandwidth_hz / (2*W)
        freqs = np.where(freqs < (f_min + eps), f_min + eps, freqs)

    # Fabrique le signal ligne par ligne
    chunks = []
    for r in range(H):
        N = samples_per_row if r < H - 1 else last_row_extra
        if N <= 0:
            continue
        t0 = (r * duration_s) / H
        # temps absolu sur la ligne (pour continuité de phase)
        t = t0 + (np.arange(N) / fs)

        # fréquences actives pour cette ligne
        active_cols = np.where(bitmap[r])[0]
        if active_cols.size == 0:
            # silence (complexe)
            row_sig = np.zeros(N, dtype=np.complex64)
        else:
            # somme des ondes complexes
            row_sig = np.zeros(N, dtype=np.complex128)  # accumulateur en float64 pour limiter le bruit d'addition
            # normalisation si beaucoup de pixels allumés
            row_amp = amp
            if normalize_per_row:
                row_amp = amp / np.sqrt(active_cols.size)

            for c in active_cols:
                f = freqs[c]
                # e^{j2π f t}
                row_sig += row_amp * np.exp(1j * 2 * np.pi * f * t)

            # fenêtre douce au bord de chaque ligne pour réduire les clicks entre lignes
            if per_row_window and N > 4:
                win = np.hanning(N)
                # aplatis un peu la fenêtre pour garder de l'énergie au centre
                # (optionnel)
                row_sig *= win

        chunks.append(row_sig.astype(np.complex64))

    iq = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.complex64)
    # Sécurité: limite l'amplitude si nécessaire (hard clip doux)
    peak = np.max(np.abs(iq)) if iq.size else 1.0
    if peak > 1.0:
        iq = (iq / peak).astype(np.complex64)
    return iq

def save_npy(iq, path):
    np.save(path, iq.astype(np.complex64))

def save_wav_stereo(iq, fs, path):
    """
    Sauvegarde en WAV 32-bit float deux canaux: I (L) et Q (R).
    Sans dépendance externe: on utilise la norme WAV simple (PCM float32).
    """
    import wave, struct
    # Normalise dans [-1,1] par sécurité
    peak = np.max(np.abs(iq)) if iq.size else 1.0
    if peak > 1.0:
        iq = iq / peak

    I = iq.real.astype(np.float32)
    Q = iq.imag.astype(np.float32)
    interleaved = np.empty((iq.size * 2,), dtype=np.float32)
    interleaved[0::2] = I
    interleaved[1::2] = Q

    with wave.open(path, 'wb') as wf:
        wf.setnchannels(2)
        wf.setsampwidth(4)  # float32
        wf.setframerate(int(fs))
        # écrire frames en bytes
        wf.writeframes(interleaved.tobytes())

def main():
    p = argparse.ArgumentParser(description="Génère un signal IQ dont la waterfall affiche un texte.")
    p.add_argument("--text", "-t", default="Hello", help="Texte à imprimer dans la waterfall.")
    p.add_argument("--fs", type=float, default=2_000_000, help="Fréquence d'échantillonnage (Hz).")
    p.add_argument("--duration", "-d", type=float, default=5.0, help="Durée totale du signal (s).")
    p.add_argument("--fmin", type=float, default=0.0, help="Fréquence basse (Hz) de la bande imprimée (souvent 0).")
    p.add_argument("--bandwidth", "-B", type=float, default=200_000.0, help="Bande passante totale (Hz) occupée par le texte.")
    p.add_argument("--font_size", type=int, default=80, help="Taille de police (avant redimensionnement).")
    p.add_argument("--font_path", type=str, default=None, help="Chemin vers un fichier .ttf / .otf (optionnel).")
    p.add_argument("--height_rows", type=int, default=256, help="Hauteur (en lignes) de l'image finale (axe temps).")
    p.add_argument("--width_cols", type=int, default=None, help="Largeur (colonnes) forcée de l'image finale (axe fréquence).")
    p.add_argument("--invert", action="store_true", help="Inverse les pixels (utile selon palette waterfall).")
    p.add_argument("--amp", type=float, default=0.8, help="Amplitude nominale des tons.")
    p.add_argument("--no_row_window", action="store_true", help="Désactive la fenêtre par ligne.")
    p.add_argument("--no_row_norm", action="store_true", help="Désactive la normalisation par nombre de tons actifs.")
    p.add_argument("--no_dc_guard", action="store_true", help="Autorise d'aller au tout début de bande (proche DC).")
    p.add_argument("--out_npy", default="spectral_text_iq.npy", help="Fichier de sortie .npy (complex64).")
    p.add_argument("--wav", default=None, help="Chemin WAV stéréo (I/Q) à écrire (optionnel).")
    p.add_argument("--reverse_time", action="store_true", help="Inverse le signal dans le temps (time reversal).")
    args = p.parse_args()

    # 1) Rendu du texte -> bitmap
    bitmap = render_text_bitmap(
        text=args.text,
        font_size=args.font_size,
        font_path=args.font_path,
        padding=10,
        invert=args.invert,
        target_width=args.width_cols,
        target_height=args.height_rows
    )

    # 2) Synthèse du signal IQ
    iq = synthesize_iq_from_bitmap(
        bitmap=bitmap,
        fs=args.fs,
        duration_s=args.duration,
        f_min=args.fmin,
        bandwidth_hz=args.bandwidth,
        amp=args.amp,
        per_row_window=(not args.no_row_window),
        normalize_per_row=(not args.no_row_norm),
        dc_guard=(not args.no_dc_guard)
    )

    if args.reverse_time:
        iq = reverse_time(iq)

    # 3) Sauvegardes
    if args.wav:
        save_wav_stereo(iq, args.fs, args.wav)
    else:
        save_npy(iq, args.out_npy)

    print(f"OK - Écrit {iq.size} échantillons IQ à fs={args.fs:.0f} Hz.")
    print(f"- NPY: {args.out_npy}")
    if args.wav:
        print(f"- WAV: {args.wav}")

if __name__ == "__main__":
    main()

# Usage
"""
python main.py -t "Hello World" --fs 2000000 --duration 5 --fmin 0 --bandwidth 200000 --height_rows 256 --wav hello.wav
"""
