#!/usr/bin/env python3
"""Cradle v1: a 'newborn' image classifier that learns from typed teaching
utterances paired with the live webcam stream.

Quickstart:
    python -m cradle.main
    # then in this terminal:
    >>> this is a cup
    >>> this is a hand
    >>> list
    >>> save
    >>> quit
"""
from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
from pathlib import Path

import cv2

from .oracle import Label, RuleBasedOracle
from .perception import Camera, CameraConfig
from .registry import LabelRegistry, RegistryFull
from .runtime import RuntimeConfig, TeacherRuntime


def stdin_reader(q: "queue.Queue[str]", stop: threading.Event) -> None:
    while not stop.is_set():
        try:
            line = sys.stdin.readline()
        except (KeyboardInterrupt, EOFError):
            stop.set()
            return
        if not line:
            stop.set()
            return
        q.put(line.rstrip("\n"))


def handle_utterance(
    text: str,
    oracle: RuleBasedOracle,
    runtime: TeacherRuntime,
    camera: Camera,
    last_pred: dict,
    ckpt_path: str,
) -> bool:
    label: Label = oracle.parse(text)

    if label.action == "noop":
        print(f'  [oracle] could not parse: "{text}". '
              f'Try: "this is a <name>", "yes"/"no", "list", "save", "quit".')
        return True

    if label.action == "command":
        if label.name == "quit":
            return False
        if label.name == "save":
            runtime.save(ckpt_path)
            print(f"  [saved] {ckpt_path} (+ .registry.pt)  steps={runtime.steps}")
            return True
        if label.name == "list":
            names = runtime.registry.known_names()
            print(f"  [classes] {len(names)}/{runtime.registry.capacity}: "
                  + (", ".join(names) if names else "(none yet)"))
            return True
        if label.name == "forget":
            print(f"  [forget] not implemented in v1 (would prune class '{label.arg}').")
            return True
        print(f"  [command] unknown: {label.name}")
        return True

    if label.action == "feedback":
        cid = last_pred.get("cid", -1)
        name = last_pred.get("name", "?")
        if label.polarity == +1 and cid >= 0:
            frames = camera.recent_tensors(runtime.cfg.frames_per_utterance)
            ran = runtime.observe(frames, cid)
            print(f"  [+] reinforced '{name}' with {len(frames)} frames"
                  f"{' (train step)' if ran else ''}")
        else:
            print("  [-] noted (negative feedback is a v2 feature).")
        return True

    if label.action == "teach":
        assert label.class_name is not None
        try:
            cid = runtime.registry.get_or_create(label.class_name)
        except RegistryFull as e:
            print(f"  [error] {e}")
            return True
        frames = camera.recent_tensors(runtime.cfg.frames_per_utterance)
        if not frames:
            print("  [warn] no frames captured yet; is the camera running?")
            return True
        ran = runtime.observe(frames, cid)
        print(f"  [teach] '{label.class_name}' (id={cid}) <- {len(frames)} frames"
              f"  buffered={runtime.buffered()}/{runtime.cfg.batch_size}"
              f"{'  ** train step **' if ran else ''}")
        return True

    return True


def draw_overlay(frame, name: str, conf: float, n_classes: int, steps: int) -> None:
    h, w = frame.shape[:2]
    text = f"{name}  conf={conf:+.2f}"
    cv2.rectangle(frame, (0, 0), (w, 32), (0, 0, 0), -1)
    cv2.putText(frame, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 255, 0) if conf > 0.5 else (0, 200, 255), 2)
    footer = f"classes={n_classes}  steps={steps}  (type utterances in the terminal)"
    cv2.rectangle(frame, (0, h - 24), (w, h), (0, 0, 0), -1)
    cv2.putText(frame, footer, (8, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (200, 200, 200), 1)


def main() -> None:
    p = argparse.ArgumentParser(description="Cradle v1: teacher-driven SPELA")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--input-size", type=int, default=96)
    p.add_argument("--num-classes", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--frames-per-utterance", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default=None, help="cuda/cpu (auto)")
    p.add_argument("--ckpt", default="cradle.ckpt")
    p.add_argument("--load", action="store_true", help="resume from --ckpt if it exists")
    p.add_argument("--no-display", action="store_true", help="headless (no cv2 window)")
    args = p.parse_args()

    registry = LabelRegistry(capacity=args.num_classes)
    runtime = TeacherRuntime(
        registry,
        RuntimeConfig(
            input_size=args.input_size,
            num_classes=args.num_classes,
            lr=args.lr,
            batch_size=args.batch_size,
            frames_per_utterance=args.frames_per_utterance,
            device=args.device,
        ),
    )

    if args.load and Path(args.ckpt).exists():
        runtime.load(args.ckpt)
        print(f"[loaded] {args.ckpt}  classes={len(registry)}  steps={runtime.steps}")

    camera = Camera(CameraConfig(device=args.camera, input_size=args.input_size))
    camera.start()
    print(f"[camera] device={args.camera} running. Device for SPELA: {runtime.device}")
    print("[hint]  Type utterances like:")
    print('         this is a cup     |   this is a hand    |    yes')
    print("         list              |   save              |    quit")

    oracle = RuleBasedOracle()
    q: "queue.Queue[str]" = queue.Queue()
    stop = threading.Event()
    threading.Thread(target=stdin_reader, args=(q, stop), name="stdin", daemon=True).start()

    last_pred = {"cid": -1, "name": "?", "conf": 0.0}
    last_predict_time = 0.0
    predict_interval = 0.1

    try:
        while not stop.is_set():
            try:
                while True:
                    text = q.get_nowait()
                    keep_going = handle_utterance(text, oracle, runtime, camera,
                                                  last_pred, args.ckpt)
                    if not keep_going:
                        stop.set()
                        break
            except queue.Empty:
                pass

            now = time.time()
            if now - last_predict_time >= predict_interval:
                t = camera.latest_tensor()
                if t is not None:
                    cid, conf, name = runtime.predict(t)
                    last_pred["cid"] = cid
                    last_pred["name"] = name
                    last_pred["conf"] = conf
                last_predict_time = now

            if not args.no_display:
                raw = camera.latest_raw()
                if raw is not None:
                    draw_overlay(raw, last_pred["name"], last_pred["conf"],
                                 len(registry), runtime.steps)
                    cv2.imshow("cradle", raw)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    stop.set()
                elif key == ord("s"):
                    runtime.save(args.ckpt)
                    print(f"  [saved] {args.ckpt}")
            else:
                time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        camera.stop()
        if not args.no_display:
            cv2.destroyAllWindows()
        print(f"[done] classes={len(registry)}  steps={runtime.steps}")


if __name__ == "__main__":
    main()
