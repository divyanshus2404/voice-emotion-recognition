# 🎙️ Voice Emotion Recognition

Speak into your mic and see the detected emotion. The app shows the emotion for the whole clip **and how it changes second by second**.
It listens to *how* you speak (tone, pitch, energy), not *what* you say.

**▶️ Live demo:** https://huggingface.co/spaces/divyanshu2404/voice-emotion-recognition  
Runs entirely in your browser: the model is exported to ONNX, so your audio never leaves your device and hosting is free.

![demo](docs/demo.gif) <!-- record a 10-second GIF of the app and put it here -->

---

## How it works (in 30 seconds)

```
 🎤 audio ──► clean up ──► wav2vec2 ──► classifier ──► 😄 happy 72%
              (mono, 16 kHz,    (pretrained on      (small layer
               trim silence,     960 h of speech)    trained on
               normalise)                            emotions)
```

1. **Clean up the audio.** Any microphone or file is converted to the same format: one channel, 16,000 samples per second, silence trimmed, volume normalised.
2. **wav2vec2 turns sound into features.** It is a transformer that learned what human speech sounds like from 960 hours of audio. I don't hand-craft features like MFCCs; the model learns them itself.
3. **A classifier picks the emotion.** A small layer on top turns those features into a probability for each emotion.
4. **Timeline.** The clip is cut into 2-second windows that slide 1 second at a time, and each window is classified. This shows mood *changes*, for example a caller going from neutral to angry.

## Why these choices

| Decision | Why |
|---|---|
| Fine-tune a pretrained model instead of training a CNN from scratch | Emotion datasets are small (RAVDESS has 1,440 clips). A model that already understands speech needs far less data. This is *transfer learning*. |
| Split the test set **by speaker**, not by random file | If the same actor appears in training and test, the model can memorise the voice and the accuracy is inflated. Actors 21–24 are never seen in training. |
| Freeze the low-level feature encoder while training | Faster training and less overfitting on a small dataset. |
| Sliding-window timeline | Real conversations aren't one emotion. This makes the output useful for things like call-center analytics. |

## Sample input → output

| Input | Output |
|---|---|
| `samples/sample_speech.wav` (computer voice: *"I can't believe you did this again…"*) | 😄 happy 51% · neutral 31% · angry 19% · sad 0% |

The computer voice is flat, so the scores are mixed. That is an honest **limitation** (see below). Record your own voice to see clearer results.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python app.py            # open http://localhost:7860
pytest                   # run the tests
```

Or with Docker:

```bash
docker build -t voice-emotion . && docker run -p 7860:7860 voice-emotion
```

## Deploy (free, runs in the browser)

The live demo is a static page in `web/`. There is no server: the browser runs the model itself with `onnxruntime-web`.

```bash
python export_onnx.py    # PyTorch -> ONNX, compressed to 8-bit (~120 MB); checks outputs match
python -m http.server 8000 -d web    # try it at http://localhost:8000
hf upload <your-username>/voice-emotion-recognition web . --repo-type space
```

`model.onnx` is too big for GitHub (100 MB limit), so it isn't committed. Run `export_onnx.py` to create it.
Only the transformer's matrix layers are compressed: compressing every layer changed the predictions too much
(sample clip: 91% happy instead of 51%). With matrix layers only, the output stays within ~2% of PyTorch.

## Train your own model (8 emotions)

By default the app uses `superb/wav2vec2-base-superb-er`: neutral, happy, angry and sad, trained on IEMOCAP.
To train on RAVDESS (neutral, calm, happy, sad, angry, fearful, disgust, surprised):

1. Download `Audio_Speech_Actors_01-24.zip` from [Zenodo](https://zenodo.org/records/1188976) and unzip it.
2. Run on a GPU (the free Google Colab tier is enough, about 15 minutes):
   ```bash
   python train.py --data_dir Audio_Speech_Actors_01-24 --epochs 10
   ```
3. It prints a per-emotion report and a confusion matrix, then saves to `./model`. `app.py` uses that folder automatically.

## Project structure

```
emotion/predictor.py   core logic: load model, clean audio, predict, timeline
app.py                 local web UI (Gradio + Plotly)
export_onnx.py         converts the model to ONNX for the browser
web/index.html         the live demo: runs the model in the browser
train.py               fine-tuning on RAVDESS with a speaker-independent split
tests/                 unit tests (probabilities sum to 1, stereo/mic input, timeline)
Dockerfile             container for deployment
```

## Limitations (and what I'd do next)

- **Acted vs. real emotion.** Datasets use actors who exaggerate. Real speech is subtler, so accuracy drops in the wild. Next: test on a different dataset (CREMA-D) to measure the drop.
- **English only**, and mostly North American accents.
- **Tone only.** Sarcasm ("great, just great") needs the words too. Next: combine with a Whisper transcript (multimodal).
