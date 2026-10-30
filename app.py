"""Web app: record or upload your voice and see the detected emotion.

Run:  python app.py   ->  open http://localhost:7860
"""
import time
from pathlib import Path

import gradio as gr
import plotly.graph_objects as go

from emotion import EMOJI, EmotionPredictor

COLORS = {
    "neutral": "#94a3b8", "calm": "#5eead4", "happy": "#facc15", "sad": "#60a5fa",
    "angry": "#f87171", "fearful": "#c084fc", "disgust": "#4ade80", "surprised": "#fb923c",
}

SAMPLES = Path(__file__).parent / "samples"
predictor = EmotionPredictor()


def result_card(emotion: str, confidence: float, seconds: float, latency_ms: float) -> str:
    color = COLORS.get(emotion, "#a78bfa")
    return f"""
    <div class="card" style="border-color:{color}">
      <div class="emoji">{EMOJI.get(emotion, "🎙️")}</div>
      <div class="label" style="color:{color}">{emotion.upper()}</div>
      <div class="meta">{confidence:.0%} confident · {seconds:.1f}s of audio · {latency_ms:.0f} ms</div>
    </div>"""


def bar_chart(scores: dict[str, float]) -> go.Figure:
    items = sorted(scores.items(), key=lambda kv: kv[1])
    fig = go.Figure(go.Bar(
        x=[v for _, v in items], y=[f"{EMOJI.get(k, '')} {k}" for k, _ in items], orientation="h",
        marker_color=[COLORS.get(k, "#a78bfa") for k, _ in items],
        text=[f"{v:.0%}" for _, v in items], textposition="outside",
    ))
    fig.update_layout(title="Emotion scores", xaxis=dict(range=[0, 1.1], tickformat=".0%"),
                      height=300, margin=dict(l=10, r=10, t=40, b=10), template="plotly_dark")
    return fig


def timeline_chart(timeline) -> go.Figure:
    fig = go.Figure()
    times = [t for t, _ in timeline]
    for label in predictor.labels:
        fig.add_trace(go.Scatter(
            x=times, y=[s[label] for _, s in timeline], name=f"{EMOJI.get(label, '')} {label}",
            mode="lines+markers", line=dict(color=COLORS.get(label), width=3, shape="spline"),
        ))
    fig.update_layout(title="Emotion over time", xaxis_title="seconds", yaxis=dict(range=[0, 1], tickformat=".0%"),
                      height=320, margin=dict(l=10, r=10, t=40, b=10), template="plotly_dark")
    return fig


def analyze(audio):
    if audio is None:
        raise gr.Error("Record or upload some audio first.")
    sr, data = audio
    seconds = len(data) / sr
    if seconds < 0.5:
        raise gr.Error("Clip is too short — speak for at least 1 second.")

    start = time.perf_counter()
    scores = predictor.predict(data, sr)
    latency_ms = (time.perf_counter() - start) * 1000
    timeline = predictor.timeline(data, sr)

    top = max(scores, key=scores.get)
    return result_card(top, scores[top], seconds, latency_ms), bar_chart(scores), timeline_chart(timeline)


CSS = """
.card {text-align:center; padding:28px; border:3px solid; border-radius:20px; background:#111827}
.card .emoji {font-size:84px; line-height:1.1}
.card .label {font-size:32px; font-weight:800; letter-spacing:3px; margin-top:6px}
.card .meta {color:#9ca3af; margin-top:8px}
#title {text-align:center}
"""

with gr.Blocks(title="Voice Emotion Recognition", css=CSS,
               theme=gr.themes.Soft(primary_hue="violet").set(body_background_fill="#0b1020")) as demo:
    gr.Markdown("# 🎙️ Voice Emotion Recognition\nSpeak into the mic or upload a clip — "
                "the model listens to **how** you say it, not **what** you say.", elem_id="title")
    with gr.Row():
        with gr.Column(scale=1):
            audio_in = gr.Audio(sources=["microphone", "upload"], type="numpy", label="Your voice")
            btn = gr.Button("Analyze emotion", variant="primary", size="lg")
            gr.Examples([str(f) for f in sorted(SAMPLES.glob("*.wav"))], audio_in, label="Try a sample")
            gr.Markdown(f"<sub>Model: `{predictor.model_name}` · emotions: {', '.join(predictor.labels)}</sub>")
        with gr.Column(scale=1):
            card = gr.HTML(result_card("neutral", 0, 0, 0).replace("NEUTRAL", "WAITING…"))
    with gr.Row():
        bars = gr.Plot()
        line = gr.Plot()

    btn.click(analyze, audio_in, [card, bars, line])
    audio_in.stop_recording(analyze, audio_in, [card, bars, line])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
