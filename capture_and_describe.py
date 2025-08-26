#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
capture_and_describe.py
-----------------------
- Cattura una foto da webcam
- Manda l'immagine a GPT Vision e ottiene una descrizione
- Converte la descrizione in audio (TTS) e la riproduce in background
- Mostra una finestra Tk con IMMAGINE + TESTO (font grande), sempre on-top
- La finestra parte grande e MEMORIZZA dimensioni/posizione per i prossimi avvii
- Chiude la finestra dopo PREVIEW_SECONDS (da .env o CLI)
- Salva immagini/audio in Desktop/CameraCaptures mantenendo gli storici
"""




import os
import sys
import json
import time
import base64
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple, Dict

# ---------- Logging ----------
def info(msg: str): print(f"[INFO] {msg}", flush=True)
def warn(msg: str): print(f"[AVVISO] {msg}", flush=True)
def err(msg: str):  print(f"[ERRORE] {msg}", flush=True)

# ---------- .env ----------
from dotenv import load_dotenv
load_dotenv()

# ---------- FFmpeg (prima di import pydub) ----------
SCRIPT_DIR = Path(__file__).resolve().parent
FFMPEG_DIR = SCRIPT_DIR / "ffmpeg_bin"

if os.name == "nt":
    # Su Windows usa i binari locali se presenti
    os.environ["PATH"] = os.environ.get("PATH", "") + os.pathsep + str(FFMPEG_DIR)
    FFMPEG_PATH = str(FFMPEG_DIR / "ffmpeg.exe")
    FFPROBE_PATH = str(FFMPEG_DIR / "ffprobe.exe")
else:
    FFMPEG_PATH = "ffmpeg"
    FFPROBE_PATH = "ffprobe"

# ---------- Librerie audio ----------
try:
    from pydub import AudioSegment
    AudioSegment.converter = FFMPEG_PATH
    AudioSegment.ffmpeg = FFMPEG_PATH
    AudioSegment.ffprobe = FFPROBE_PATH
except Exception as e:
    warn(f"Pydub/FFmpeg non configurato: {e}")

try:
    import simpleaudio as sa
except Exception as e:
    sa = None
    warn(f"simpleaudio non disponibile: {e}")

WINSOUND_OK = False
if os.name == "nt":
    try:
        import winsound
        WINSOUND_OK = True
    except Exception:
        WINSOUND_OK = False

# ---------- OpenAI SDK ----------
try:
    from openai import OpenAI
except ImportError:
    err("Libreria 'openai' mancante. Installa con: pip install openai")
    sys.exit(1)

# ---------- OpenCV ----------
try:
    import cv2
except ImportError:
    err("Libreria 'opencv-python' mancante. Installa con: pip install opencv-python")
    sys.exit(1)

# ---------- Tkinter + Pillow ----------
TK_OK = True
try:
    import tkinter as tk
    from tkinter import ttk
    from PIL import Image, ImageTk
    import tkinter.font as tkFont
except Exception as e:
    TK_OK = False
    warn(f"UI Tk disabilitata (serve tkinter + pillow): {e}")


# ======================
# Utility percorsi
# ======================

def get_output_dir() -> Path:
    home = Path.home()
    desktop_candidates = [home / "Desktop", home / "Scrivania"]
    desktop = next((p for p in desktop_candidates if p.exists()), home)
    out_dir = desktop / "CameraCaptures"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def b64_image(image_path: Path) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ======================
# Audio helpers
# ======================

def process_audio(input_path: Path, output_path: Path, speed: float = 1.0, gain_db: float = 0.0):
    """Applica speed e gain all'audio (richiede ffmpeg). Output in MP3."""
    if speed <= 1.0 and abs(gain_db) < 0.1:
        return
    try:
        seg = AudioSegment.from_file(input_path)
        if speed > 1.0:
            seg = seg._spawn(seg.raw_data, overrides={
                "frame_rate": int(seg.frame_rate * speed)
            }).set_frame_rate(seg.frame_rate)
        if abs(gain_db) >= 0.1:
            seg = seg + gain_db
        seg.export(output_path, format="mp3")
        info(f"Audio processato: speed={speed}x, gain={gain_db:+.1f}dB")
    except Exception as e:
        warn(f"Processing audio fallito: {e}")


def _play_with_simpleaudio(mp3_path: Path):
    """Decodifica con pydub e suona via simpleaudio (non blocca)."""
    if sa is None:
        return None, None
    seg = AudioSegment.from_file(mp3_path)
    play_obj = sa.play_buffer(seg.raw_data, num_channels=seg.channels,
                              bytes_per_sample=seg.sample_width, sample_rate=seg.frame_rate)
    return play_obj, seg  # tenere riferimenti vivi


def _play_with_winsound_fallback(mp3_path: Path, temp_dir: Path):
    """Su Windows: converti a WAV e suona con winsound (nessuna finestra)."""
    if not WINSOUND_OK:
        return None
    try:
        wav_path = temp_dir / (mp3_path.stem + ".wav")
        AudioSegment.from_file(mp3_path).export(wav_path, format="wav")
        winsound.PlaySound(str(wav_path), winsound.SND_FILENAME | winsound.SND_ASYNC)
        return wav_path
    except Exception as e:
        warn(f"Fallback winsound fallito: {e}")
        return None


def play_audio_background(mp3_path: Path, temp_dir: Path) -> Dict[str, object]:
    """
    Prova simpleaudio; su Windows fallback a winsound (WAV).
    Ritorna refs per evitare garbage-collection.
    """
    try:
        play_obj, seg = _play_with_simpleaudio(mp3_path)
        if play_obj:
            info("Audio: riproduzione via simpleaudio.")
            return {"play_obj": play_obj, "seg": seg, "wav_path": None}
        if os.name == "nt":
            wav_path = _play_with_winsound_fallback(mp3_path, temp_dir)
            if wav_path:
                info("Audio: fallback winsound (WAV).")
                return {"play_obj": None, "seg": None, "wav_path": wav_path}
        warn("Nessun backend audio disponibile.")
        return {"play_obj": None, "seg": None, "wav_path": None}
    except Exception as e:
        warn(f"Riproduzione audio fallita: {e}")
        return {"play_obj": None, "seg": None, "wav_path": None}


# ======================
# UI (Tk): immagine + testo, memorizza dimensione
# ======================

def load_ui_geometry(state_path: Path) -> Optional[str]:
    try:
        if state_path.exists():
            data = json.loads(state_path.read_text(encoding="utf-8"))
            return data.get("geometry")
    except Exception:
        pass
    return None


def save_ui_geometry(state_path: Path, geometry: str):
    try:
        state_path.write_text(json.dumps({"geometry": geometry}), encoding="utf-8")
    except Exception as e:
        warn(f"Salvataggio UI state fallito: {e}")


def show_image_and_text_blocking(
    image_path: Path,
    text_answer: str,
    seconds: int,
    title: str,
    font_family: str,
    font_size: int,
    ui_state_path: Path
):
    if not TK_OK:
        warn("Tkinter non disponibile: salto UI.")
        return

    root = tk.Tk()
    root.title(f"{title} ({seconds}s)")
    try:
        root.attributes("-topmost", True)
    except Exception:
        pass

    # Tema
    try:
        style = ttk.Style()
        style.theme_use("clam")
    except Exception:
        pass

    # Font
    big_font = tkFont.Font(family=font_family, size=font_size)
    small_font = tkFont.Font(family=font_family, size=max(10, font_size - 6))

    # Container
    container = ttk.Frame(root, padding=12)
    container.pack(fill="both", expand=True)

    # Immagine ridimensionata (max ~55% larghezza schermo / 80% altezza)
    img = Image.open(image_path)
    sw = root.winfo_screenwidth()
    sh = root.winfo_screenheight()
    max_w = int(sw * 0.55)
    max_h = int(sh * 0.80)
    w, h = img.size
    scale = min(max_w / w, max_h / h, 1.0)
    img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    tk_img = ImageTk.PhotoImage(img)

    # Layout
    container.columnconfigure(0, weight=0)
    container.columnconfigure(1, weight=1)
    container.rowconfigure(0, weight=1)

    # Colonna immagine
    img_frame = ttk.Frame(container, padding=(0, 0, 12, 0))
    img_frame.grid(row=0, column=0, sticky="nsew")
    ttk.Label(img_frame, image=tk_img).pack()

    # Colonna testo con scrollbar
    text_frame = ttk.Frame(container)
    text_frame.grid(row=0, column=1, sticky="nsew")
    text_frame.columnconfigure(0, weight=1)
    text_frame.rowconfigure(0, weight=1)

    text_widget = tk.Text(text_frame, wrap="word", height=20,
                          font=big_font, padx=14, pady=14)
    text_widget.insert("1.0", text_answer)
    text_widget.config(state="disabled")
    text_widget.grid(row=0, column=0, sticky="nsew")

    scroll = ttk.Scrollbar(text_frame, orient="vertical", command=text_widget.yview)
    scroll.grid(row=0, column=1, sticky="ns")
    text_widget.config(yscrollcommand=scroll.set)

    # Barra inferiore
    bottom = ttk.Frame(root, padding=(12, 8, 12, 12))
    bottom.pack(fill="x")
    countdown_label = ttk.Label(bottom, text=f"Chiusura automatica tra {seconds}s", font=small_font)
    countdown_label.pack(side="left")
    ttk.Button(bottom, text="Chiudi ora", command=lambda: on_close()).pack(side="right")

    # Geometria iniziale: ultima usata oppure 80% schermo
    def center_default():
        init_w = int(sw * 0.8)
        init_h = int(sh * 0.8)
        x = (sw - init_w) // 2
        y = (sh - init_h) // 2
        return f"{init_w}x{init_h}+{x}+{y}"

    last_geo = load_ui_geometry(ui_state_path)
    root.geometry(last_geo if last_geo else center_default())

    # Chiusura con salvataggio geometria
    def on_close():
        try:
            save_ui_geometry(ui_state_path, root.winfo_geometry())
        finally:
            try:
                root.destroy()
            except Exception:
                pass

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.bind("<Escape>", lambda e: on_close())

    # Countdown
    remaining = {"sec": seconds}
    def tick():
        remaining["sec"] -= 1
        if remaining["sec"] <= 0:
            on_close()
            return
        root.title(f"{title} ({remaining['sec']}s)")
        countdown_label.config(text=f"Chiusura automatica tra {remaining['sec']}s")
        root.after(1000, tick)

    root.after(seconds * 1000, on_close)
    root.after(1000, tick)

    # Mainloop bloccante
    root.mainloop()


# ======================
# Webcam
# ======================

def enumerate_cameras(max_indices: int = 10) -> List[Tuple[int, str]]:
    found = []
    if os.name == "nt":
        backends = [(cv2.CAP_DSHOW, "DirectShow"), (cv2.CAP_MSMF, "MediaFoundation"), (cv2.CAP_ANY, "Auto")]
    else:
        backends = [(cv2.CAP_V4L2, "V4L2"), (cv2.CAP_ANY, "Auto")]
    for i in range(max_indices + 1):
        for be, name in backends:
            cap = cv2.VideoCapture(i, be)
            if cap.isOpened():
                ok, _ = cap.read()
                cap.release()
                if ok:
                    found.append((i, name))
                    break
    uniq = {}
    for idx, name in found:
        if idx not in uniq:
            uniq[idx] = name
    return [(k, v) for k, v in sorted(uniq.items(), key=lambda x: x[0])]


def open_camera_auto(device: str, width: int, height: int) -> Tuple[cv2.VideoCapture, int, str]:
    if os.name == "nt":
        os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_MSMF", "0")
        backend_list = [(cv2.CAP_DSHOW, "DirectShow"), (cv2.CAP_MSMF, "MediaFoundation"), (cv2.CAP_ANY, "Auto")]
    else:
        backend_list = [(cv2.CAP_V4L2, "V4L2"), (cv2.CAP_ANY, "Auto")]

    if device.lower() == "auto":
        indices = list(range(0, 11))
    else:
        try:
            idx = int(device)
        except ValueError:
            raise RuntimeError(f"--device deve essere 'auto' oppure un numero. Ricevuto: {device}")
        indices = [idx]

    for i in indices:
        for be, name in backend_list:
            cap = cv2.VideoCapture(i, be)
            if cap.isOpened():
                time.sleep(0.2)
                ok, _ = cap.read()
                if ok:
                    if width > 0: cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                    if height > 0: cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                    return cap, i, name
                cap.release()
    raise RuntimeError("Impossibile aprire la webcam su nessun backend/indice.")


def capture_webcam_image(device: str, width: int, height: int, out_path: Path) -> Path:
    cap, used_index, backend_name = open_camera_auto(device, width, height)
    info(f"Webcam aperta su index={used_index} backend={backend_name}")
    time.sleep(0.3)
    for _ in range(5):
        cap.read()
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        raise RuntimeError("Acquisizione frame fallita dalla webcam.")
    ok = cv2.imwrite(str(out_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not ok:
        raise RuntimeError(f"Salvataggio immagine fallito: {out_path}")
    return out_path


# ======================
# OpenAI Vision + TTS
# ======================

def gpt_vision_describe(client: OpenAI, model_vision: str, prompt: str, image_path: Path) -> str:
    try:
        img_b64 = b64_image(image_path)
        data_url = f"data:image/jpeg;base64,{img_b64}"
        completion = client.chat.completions.create(
            model=model_vision,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }],
            temperature=0.2,
        )
        return (completion.choices[0].message.content or "").strip()
    except Exception as e:
        err(f"Vision API fallita: {e}")
        return "Errore nella descrizione dell'immagine."


def gpt_tts_generate(client: OpenAI, model_tts: str, voice: str, text: str, out_mp3: Path):
    try:
        resp = client.audio.speech.create(model=model_tts, voice=voice, input=text)
        with open(out_mp3, "wb") as f:
            f.write(resp.content)
        info(f"Audio TTS salvato: {out_mp3}")
    except Exception as e:
        err(f"TTS API fallita: {e}")
        raise


# ======================
# Main
# ======================

def main():
    parser = argparse.ArgumentParser(description="Cattura, descrivi con GPT Vision, parla con TTS, mostra finestra grande.")
    parser.add_argument("--device", type=str, default=os.getenv("CAM_DEVICE", "auto"), help="Indice webcam (0,1,2) o 'auto'.")
    parser.add_argument("--width", type=int, default=int(os.getenv("CAM_WIDTH", "1280")), help="Larghezza desiderata")
    parser.add_argument("--height", type=int, default=int(os.getenv("CAM_HEIGHT", "720")), help="Altezza desiderata")
    parser.add_argument("--preview_seconds", type=int, default=int(os.getenv("PREVIEW_SECONDS", "30")), help="Durata finestra (s)")
    parser.add_argument("--text_font_family", type=str, default=os.getenv("TEXT_FONT_FAMILY", "Helvetica"), help="Font testo")
    parser.add_argument("--text_font_size", type=int, default=int(os.getenv("TEXT_FONT_SIZE", "18")), help="Dimensione font")
    parser.add_argument("--prompt", type=str, default=os.getenv("VISION_PROMPT", "Descrivi in modo chiaro e utile cosa vedi nell'immagine."), help="Prompt Vision")
    parser.add_argument("--model_vision", type=str, default=os.getenv("MODEL_VISION", "gpt-4o-mini"), help="Modello Vision")
    parser.add_argument("--model_tts", type=str, default=os.getenv("MODEL_TTS", "gpt-4o-mini-tts"), help="Modello TTS")
    parser.add_argument("--voice", type=str, default=os.getenv("VOICE", "breeze"), help="Voce TTS")
    parser.add_argument("--voice_speed", type=float, default=float(os.getenv("VOICE_SPEED", "1.5")), help="Velocità audio (>1 più veloce)")
    parser.add_argument("--voice_gain_db", type=float, default=float(os.getenv("VOICE_GAIN_DB", "0")), help="Guadagno audio in dB")
    parser.add_argument("--list", action="store_true", help="Elenca webcam disponibili e termina.")
    parser.add_argument("--skip_gpt", action="store_true", help="Salta Vision (test UI/audio).")
    parser.add_argument("--skip_tts", action="store_true", help="Salta TTS (solo UI).")
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key and not args.skip_gpt:
        err("OPENAI_API_KEY mancante nel .env")
        sys.exit(1)

    if args.list:
        cams = enumerate_cameras(10)
        if not cams:
            print("Nessuna webcam rilevata.")
        else:
            print("Webcam disponibili:")
            for idx, be in cams:
                print(f" - index {idx} (backend: {be})")
        return

    client = None if args.skip_gpt else OpenAI(api_key=api_key)
    out_dir = get_output_dir()
    ui_state_path = out_dir / "ui_state.json"

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    img_path = out_dir / f"capture_{ts}.jpg"
    mp3_path = out_dir / f"speech_{ts}.mp3"

    # 1) Scatta
    info("Acquisizione immagine...")
    try:
        image_file = capture_webcam_image(args.device, args.width, args.height, img_path)
        info(f"Immagine salvata: {image_file}")
    except Exception as e:
        err(str(e)); sys.exit(1)

    # 2) Vision
    if args.skip_gpt:
        text_answer = "Vision disattivato (--skip_gpt). Test UI/audio."
    else:
        info("Invio a GPT Vision...")
        text_answer = gpt_vision_describe(client, args.model_vision, args.prompt, image_file)

    # 3) TTS
    audio_refs = {"play_obj": None, "seg": None, "wav_path": None}
    if not args.skip_tts:
        info("Generazione TTS...")
        try:
            gpt_tts_generate(client, args.model_tts, args.voice, text_answer, mp3_path)
            # speed/gain (se impostati)
            if args.voice_speed > 1.0 or abs(args.voice_gain_db) >= 0.1:
                process_audio(mp3_path, mp3_path, speed=args.voice_speed, gain_db=args.voice_gain_db)
            info("Riproduzione audio in background...")
            audio_refs = play_audio_background(mp3_path, out_dir)
        except Exception:
            warn("Audio non riprodotto per errore TTS.")

    # 4) UI (bloccante)
    if args.preview_seconds > 0:
        show_image_and_text_blocking(
            image_path=image_file,
            text_answer=text_answer,
            seconds=args.preview_seconds,
            title="Scatto & Descrizione",
            font_family=args.text_font_family,
            font_size=args.text_font_size,
            ui_state_path=ui_state_path
        )
    else:
        info("UI disattivata (PREVIEW_SECONDS=0).")

    info("Fatto.")
    print(f" - Immagine: {image_file}")
    if not args.skip_tts:
        print(f" - Audio   : {mp3_path}")


if __name__ == "__main__":
    # Output non bufferizzato
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    main()
