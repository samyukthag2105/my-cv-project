# Offline Face Check-In with Liveness Detection

A small computer vision project that recognises a person's face from a webcam, asks them to blink to prove they are a real human (and not a photo), and then logs them in — all running offline on a Windows laptop, no internet needed.

This is my first proper CV project. I built it phase by phase for the NHAI Hackathon 7.0, and honestly I am still learning, but every piece in here is something I can explain in my own words now. There is a long way to go, but these were the baby steps.

## Why it exists

A lot of attendance and check-in systems can be fooled by just holding up a photo of someone. That felt like a real gap to me. So the idea was: what if the system could tell the difference between a *real* person and a *picture* of that person? A real person can blink. A photo cannot. That one simple difference is the whole heart of this project.

It is meant for places where you need a quick, low-cost, offline way to confirm "yes, this real person was actually here" — without sending anyone's face to a cloud server.

## How to run it

You need **Python 3.12** on Windows (not 3.13 — the vision libraries do not have a 3.13 version yet, I found this out the hard way).

Open Command Prompt and run these in order:

```
cd my-cv-project
python -m venv venv
venv\Scripts\activate
```

Your prompt should now start with `(venv)`. Then install everything:

```
pip install opencv-contrib-python
pip install mediapipe
pip install pyttsx3
```

Now run the parts in order. First teach it your face (it takes 30 pictures, so turn your head slightly):

```
python enroll.py
```

Then train the model on those pictures:

```
python train.py
```

Then do the full check-in (recognise → blink → log):

```
python checkin.py
```

Press **Q** to quit any window, and **R** to reset the check-in and try again.

> Note: `venv` and the `dataset` folder of face photos are **not** uploaded to GitHub on purpose. So after you clone this repo you have to make your own venv and run `enroll.py` to add your own face before it will recognise you.

## How it works

The whole thing is a pipeline of four steps:

1. **Detect** — OpenCV finds where a face is in the camera frame and draws a box around it.
2. **Recognise** — an LBPH recogniser turns the face into a number pattern and checks how close it is to the faces it was trained on. Close enough = it knows who you are.
3. **Liveness (the important bit)** — MediaPipe puts a fine mesh of dots on the face and measures the Eye Aspect Ratio (EAR). When you blink, your eye height collapses and the EAR drops, then bounces back. Catching that drop-and-recover is how it knows you actually blinked. A photo's eyes never change, so a photo can never pass this step.
4. **Log** — once you are recognised *and* you blink, your name and the time are written to `attendance.csv`, and the laptop says "thank you, you are checked in" out loud (offline voice).

## Limitations

I want to be honest about what this is and is not, because pretending it is perfect would be silly.

- **Blink liveness can be fooled by a video.** If someone plays a video of you blinking instead of a still photo, the current version would not catch it. A photo is stopped; a video is not. This is the biggest weakness and the first thing I would fix next.
- It is a **prototype**, not a real security product. Lighting changes, glasses, or a very different camera can throw off the recognition.
- The face recognition (LBPH) is a simple, classic method. It works great for a small number of enrolled people but would not scale well to hundreds of faces.
- The blink and recognition thresholds are tuned by hand, so on a different laptop they might need adjusting.

These are not things I am hiding — they are the roadmap for where this goes next.

## Demo

*(I will add a short screen recording here showing a normal check-in working, and then a photo being held up and getting refused because it cannot blink — which is the moment that proves the whole idea.)*

---

Built by Samyuktha Ganesan as a learning project. One rung at a time.
