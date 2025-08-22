# Spectral Text Printer — génère un signal IQ (complexe) dont la waterfall affiche un texte.

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
