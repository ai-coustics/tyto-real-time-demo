// Tyto web demo, browser client.
//
// Thin client: capture the mic, stream PCM16 to the Python backend, play the
// agent audio it streams back, and render the page from the backend's messages.
// All scoring, the three adaptation layers, Jev's verdicts, and the keys live on
// the backend. The look follows the ai-coustics design system (tokens in /ds)
// and the Audio Insight post-call demo: mono eyebrows, display numerals,
// hairline cards, thin severity bars.

const $ = (id) => document.getElementById(id);

const SAMPLE_RATE = 24000;       // PCM16 mono, matches the backend and OpenAI
const MIC_CHUNK = 480;           // ~20 ms batches sent to the backend

// The six Tyto 1.1 dimensions and their thresholds mirror src/tyto_voice/decision.py.
const ENV_KEYS = ["noise", "speaker_reverb", "speaker_loudness", "interfering_speech", "packet_loss", "codec_degradation"];
const SHORT = {
  noise: "Noise", speaker_reverb: "Reverb", speaker_loudness: "Loudness",
  interfering_speech: "Interf. speech", packet_loss: "Packet loss", codec_degradation: "Codec",
};
const DESCRIPTIONS = {
  noise: "Ambient noise behind the speaker, relative to the speaker's level.",
  speaker_reverb: "Low = dry, near-field audio; high = reverberant, far-field. Informational only.",
  speaker_loudness: "Loudness level of the main speaker. Informational only.",
  interfering_speech: "Competing speech behind the main speaker: people nearby, or a TV, radio or phone.",
  packet_loss: "Audio dropouts or discontinuities: packet loss, jitter, frame erasure, CPU overload.",
  codec_degradation: "Compression artifacts from the codec carrying the audio.",
};
const NO_POLARITY = new Set(["speaker_loudness", "speaker_reverb"]);
const THRESHOLDS = {
  noise: [0.20, 0.45], interfering_speech: [0.15, 0.35], packet_loss: [0.05, 0.15],
  codec_degradation: [0.30, 0.50], speaker_reverb: [0.25, 0.55], speaker_loudness: [0.12, 0.25],
};
const COMPOSITE_TH = [0.30, 0.50];   // Tyto risk bands from the docs
const NUDGE_TH = { min: 0.30, max: 0.50, default: 0.40 };

// Traffic light from the secondary palette, as in the post-call demo.
const TONE = { good: "var(--teal)", warn: "var(--amber)", bad: "var(--clay)", neutral: "var(--gray-60)" };
const band = (v, th = COMPOSITE_TH) => (v < th[0] ? "good" : v <= th[1] ? "warn" : "bad");
const dimBand = (k, v) => (NO_POLARITY.has(k) ? "neutral" : band(v, THRESHOLDS[k] || COMPOSITE_TH));
const fmt = (v) => (v == null || !Number.isFinite(v) ? "–" : v.toFixed(2));
const ACTION_WORDS = {
  ask_now: "Cut in now", ask_after_sentence: "After this sentence", adapt_quietly: "Carry on, carefully", stay_silent: "Stay quiet",
};

// ── rendering ────────────────────────────────────────────────────────────────
function setStatus(state, label) {
  const tone = { live: "positive", connecting: "warning", error: "critical" }[state] || "";
  const [main, ...detail] = String(label).split(" · ");   // "Live · gpt-live-1": phones show only "Live"
  const el = $("status");
  el.className = "badge " + tone; el.title = label;
  el.innerHTML = '<span class="dot"></span><span class="label"></span><span class="detail"></span>';
  el.querySelector(".label").textContent = main;
  el.querySelector(".detail").textContent = detail.length ? " · " + detail.join(" · ") : "";
}
function setTytoState(state, text) {
  const el = $("tyto-state");
  el.className = "state " + (state || "");
  el.textContent = text;
}
function buildDims() {
  const grid = $("env-grid"); grid.innerHTML = "";
  for (const k of ENV_KEYS) {
    const c = document.createElement("div"); c.className = "dim"; c.title = DESCRIPTIONS[k];
    c.innerHTML = `<div class="dim-label">${SHORT[k]}</div><div class="dim-val" id="val-${k}">–</div>` +
      `<div class="bar"><div class="bar-fill" id="fill-${k}"></div></div>`;
    grid.appendChild(c);
  }
}
function renderDims(m) {
  for (const k of ENV_KEYS) {
    const v = m[k]; if (v == null) continue;
    const color = TONE[dimBand(k, v)];
    const val = $(`val-${k}`), fill = $(`fill-${k}`);
    val.textContent = fmt(v);
    val.style.color = NO_POLARITY.has(k) ? "var(--text-secondary)" : color;
    fill.style.width = Math.max(2, Math.min(100, v * 100)).toFixed(1) + "%";
    fill.style.background = color;
  }
}
function renderRisk(v) {
  if (v == null) return;
  const b = band(v);
  $("risk-val").textContent = fmt(v); $("risk-val").style.color = TONE[b];
  $("risk-band").textContent = b;
  $("risk-fill").style.width = Math.max(2, Math.min(100, v * 100)).toFixed(1) + "%";
  $("risk-fill").style.background = TONE[b];
}
function resetRisk() {
  $("risk-val").textContent = "–"; $("risk-val").style.color = ""; $("risk-band").textContent = "";
  $("risk-fill").style.width = "2%"; $("risk-fill").style.background = "";
  for (const k of ENV_KEYS) { $(`val-${k}`).textContent = "–"; $(`val-${k}`).style.color = ""; $(`fill-${k}`).style.width = "2%"; }
}
const setAgent = (value, sub) => { $("agent-val").textContent = value; $("agent-sub").textContent = sub || ""; };
const setJudge = (value, sub) => { $("judge-val").textContent = value; $("judge-sub").textContent = sub || ""; };

// conversation: one list, the user, the agent, and Tyto's interventions
const convo = { lines: [], interim: { user: "", agent: "" } };
function lineEl(who, text, cls) {
  const d = document.createElement("div"); d.className = `line ${who} ${cls || ""}`;
  const label = who === "user" ? "You" : who === "agent" ? "Agent" : "Tyto";
  d.innerHTML = `<span class="who">${label}</span><span class="what"></span>`;
  d.querySelector(".what").textContent = text;
  return d;
}
function addLine(who, text, cls) {
  if (text) {
    convo.lines.push({ who, text, cls });
    if (convo.lines.length > 80) convo.lines.shift();
  }
  renderConvo();
}
function renderConvo() {
  const el = $("conversation"); el.innerHTML = "";
  for (const l of convo.lines) el.appendChild(lineEl(l.who, l.text, l.cls));
  for (const who of ["user", "agent"]) if (convo.interim[who]) el.appendChild(lineEl(who, convo.interim[who], "interim"));
  if (!el.children.length) el.innerHTML = '<div class="empty">The conversation shows up here.</div>';
  el.scrollTop = el.scrollHeight;
}
function clearConvo() { convo.lines = []; convo.interim = { user: "", agent: "" }; renderConvo(); }

// activity: the backend's event log, one badge per kind
function toneFor(kind) {
  if (kind.startsWith("tyto.nudge")) return "warning";
  if (kind.startsWith("jev.") || kind.startsWith("tool.")) return "accent";
  if (kind === "error" || kind === "tyto.error") return "critical";
  return "";
}
function log(kind, text) {
  const $log = $("log");
  const empty = $log.querySelector(".empty"); if (empty) empty.remove();
  const ts = new Date().toTimeString().slice(0, 8);
  const e = document.createElement("div"); e.className = "entry";
  e.innerHTML = `<span class="ts">${ts}</span><span class="badge sm ${toneFor(kind)}">${kind}</span><span class="msg"></span>`;
  e.querySelector(".msg").textContent = text || ""; e.querySelector(".msg").title = text || "";
  $log.appendChild(e);
  while ($log.children.length > 200) $log.removeChild($log.firstChild);
  $log.scrollTop = $log.scrollHeight;
}

// nudge sensitivity slider (tells the backend)
function setNudgeThreshold(v) {
  const val = Math.min(NUDGE_TH.max, Math.max(NUDGE_TH.min, v));
  $("nudge-th").value = val.toFixed(2);
  $("nudge-th-val").textContent = `≥ ${val.toFixed(2)}`;
  $("risk-marker").style.left = (val * 100).toFixed(1) + "%";
  $("risk-sub").textContent = `lower is better · nudge at ≥ ${val.toFixed(2)}`;
  send({ type: "nudge_threshold", value: val });
}

// ── Transport: websocket + mic + agent playback ───────────────────────────────
let ws = null, connected = false;
let micCtx = null, micStream = null, tapNode = null, micBuf = [];
let playCtx = null, playHead = 0, agentDone = false, agentPlaying = false;
const sources = new Set();  // scheduled agent audio, so a flush can stop it
let lastRoom = null, lastVad = null, lastJev = null;
let endReason = "";  // why the server ended the call (busy, time cap), shown after hang-up

function send(obj) { if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(obj)); }

const MIC_WORKLET = `
class Tap extends AudioWorkletProcessor{
  process(inputs){ const ch=inputs[0][0]; if(ch) this.port.postMessage(new Float32Array(ch)); return true; }
}
registerProcessor("tap",Tap);`;

async function start() {
  setStatus("connecting", "Connecting");
  endReason = "";
  clearConvo(); resetRisk(); lastRoom = null; lastVad = null; lastJev = null;
  setAgent("Connecting", "calling the agent"); setJudge("–", "Jev picks the agent's move when audio gets bad");
  try {
    micStream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: false, autoGainControl: false },
    });
    micCtx = new AudioContext({ sampleRate: SAMPLE_RATE });
    if (micCtx.state === "suspended") await micCtx.resume();
    playCtx = new AudioContext({ sampleRate: SAMPLE_RATE });
    if (playCtx.state === "suspended") await playCtx.resume();

    ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);
    ws.binaryType = "arraybuffer";
    ws.onopen = () => { send({ type: "start" }); };
    ws.onmessage = onMessage;
    ws.onclose = () => stop();
    ws.onerror = () => { setStatus("error", "Error"); };

    const url = URL.createObjectURL(new Blob([MIC_WORKLET], { type: "application/javascript" }));
    await micCtx.audioWorklet.addModule(url);
    const src = micCtx.createMediaStreamSource(micStream);
    tapNode = new AudioWorkletNode(micCtx, "tap");
    tapNode.port.onmessage = (e) => pushMic(e.data);
    src.connect(tapNode);

    connected = true;
    $("mic").classList.add("live"); $("mic-label").textContent = "Hang up";
    log("ws.open", "connected to backend");
  } catch (err) {
    setStatus("error", "Error"); log("error", String(err)); stop();
  }
}

function stop() {
  if (!connected && !ws) return;
  connected = false;
  $("mic").classList.remove("live"); $("mic-label").textContent = "Talk to the agent";
  send({ type: "stop" });
  if (ws) { try { ws.close(); } catch {} ws = null; }
  if (tapNode) { try { tapNode.disconnect(); } catch {} tapNode = null; }
  if (micStream) { micStream.getTracks().forEach((t) => t.stop()); micStream = null; }
  if (micCtx) { micCtx.close().catch(() => {}); micCtx = null; }
  flushPlayback();
  if (playCtx) { playCtx.close().catch(() => {}); playCtx = null; }
  setTytoState(null, "start to score your mic");
  setAgent("Idle", "what it does about your audio");
  setStatus("", endReason || "Not connected");
}

// mic: batch ~20 ms of float32 into PCM16 and send
function pushMic(chunk) {
  for (let i = 0; i < chunk.length; i++) micBuf.push(chunk[i]);
  while (micBuf.length >= MIC_CHUNK) {
    const slice = micBuf.splice(0, MIC_CHUNK);
    const pcm = new Int16Array(MIC_CHUNK);
    for (let i = 0; i < MIC_CHUNK; i++) pcm[i] = Math.max(-1, Math.min(1, slice[i])) * 32767;
    if (ws && ws.readyState === WebSocket.OPEN) ws.send(pcm.buffer);
  }
}

// agent playback: schedule each PCM16 chunk back to back
function playChunk(arrayBuffer) {
  if (!playCtx) return;
  const pcm = new Int16Array(arrayBuffer);
  const buf = playCtx.createBuffer(1, pcm.length, SAMPLE_RATE);
  const data = buf.getChannelData(0);
  for (let i = 0; i < pcm.length; i++) data[i] = pcm[i] / 32768;
  const node = playCtx.createBufferSource();
  node.buffer = buf; node.connect(playCtx.destination);
  const now = playCtx.currentTime;
  if (playHead < now) playHead = now;
  node.start(playHead);
  playHead += buf.duration;
  sources.add(node);
  if (!agentPlaying) { agentPlaying = true; send({ type: "agent_playing", value: true }); }
  node.onended = () => { sources.delete(node); maybeIdle(); };
}
function maybeIdle() {
  if (sources.size === 0 && agentDone && agentPlaying) {
    agentPlaying = false; agentDone = false;
    send({ type: "agent_playing", value: false });
  }
}
function flushPlayback() {
  agentDone = false;
  if (agentPlaying) { agentPlaying = false; send({ type: "agent_playing", value: false }); }
  // Stop what is already scheduled, or an interrupt only resets counters while the
  // queued audio keeps playing. Detach onended first so a stopped node cannot
  // touch the bookkeeping of the next response.
  for (const node of sources) { node.onended = null; try { node.stop(); } catch {} node.disconnect(); }
  sources.clear(); playHead = 0;
}

// the Agent tile: the latest adaptation, Reactive over Aware over Tuned
function onLayers(room, vad) {
  if (vad !== lastVad) {
    lastVad = vad;
    if (vad === "patient") setAgent("Waiting longer", "noisy room: it lets you finish before answering");
    else if (lastRoom !== null) setAgent("Listening", "quiet again");
  }
  if (room !== lastRoom) {
    lastRoom = room;
    if (room) setAgent("Knows the room", "told: " + room.replace(/^Audio note:\s*/i, "").split(". ")[0]);
    else setAgent("Listening", "room sounds clean");
  }
}

function onMessage(ev) {
  if (ev.data instanceof ArrayBuffer) { playChunk(ev.data); return; }
  const m = JSON.parse(ev.data);
  switch (m.type) {
    case "status":
      setStatus(m.state, m.label);
      if (m.state === "live") setAgent("Listening", "the agent says hello");
      break;
    case "config":
      $("models-val").textContent = m.backend.startsWith("gpt-live") ? "GPT-Live 1" : m.backend;
      $("models-sub").textContent = `scored by Tyto 1.1` + (m.judge ? ` · judged by Jev` : "");
      setJudge(m.judge ? "Ready" : "Rule", m.judge ? "Jev picks the agent's move when audio gets bad" : "no Jev key: a fixed rule asks you to fix it");
      log("config", `voice ${m.backend}, judge ${m.judge || "off"}`); break;
    case "tyto_state":
      if (m.state === "loading") setTytoState("warming", "loading Tyto");
      else if (m.state === "warming") setTytoState("warming", m.text || "warming up, keep talking");
      else if (m.state === "live") setTytoState(null, "live · lower is better");
      else if (m.state === "error") setTytoState("error", m.text || "Tyto error");
      log(`tyto.${m.state}`, m.text || ""); break;
    case "scores":
      renderRisk(m.scores.risk_score); renderDims(m.scores); onLayers(m.room || "", m.vad || "eager"); break;
    case "transcript":
      // Clear the grey in-progress copy before drawing the finished line, or both stay on
      // screen until the next word arrives.
      if (m.final) { convo.interim[m.who] = ""; addLine(m.who, m.text); }
      else { convo.interim[m.who] += m.text; renderConvo(); }
      break;
    case "jev":
      lastJev = m;
      setJudge(ACTION_WORDS[m.action] || m.action,
        m.source === "fallback" ? `rule fallback: ${m.reason}` : `${m.reason} · ${Math.round((m.confidence || 0) * 100)}% · ${m.latency_ms} ms`);
      log("jev.decision", `${m.action} p=${(m.confidence || 0).toFixed(2)} ${m.latency_ms} ms: ${m.reason}`); break;
    case "nudge":
      setAgent("Asking you to fix it", m.text);
      addLine("tyto", `${m.label} ${m.value.toFixed(2)}` + (lastJev ? ` · Jev: ${lastJev.source === "fallback" ? "rule" : `${(ACTION_WORDS[lastJev.action] || lastJev.action).toLowerCase()} (${Math.round((lastJev.confidence || 0) * 100)}%)`}` : "") + ` · the agent stops and asks you to fix it`, "action");
      lastJev = null;
      log("tyto.nudge", m.text); break;
    case "ended": endReason = m.text; log("session.ended", m.text); break;
    case "agent_done": agentDone = true; maybeIdle(); break;
    case "flush": flushPlayback(); break;
    case "log":
      log(m.kind, m.text);
      if (m.kind === "tyto.input.resumed") setAgent("Listening", "back to the conversation");
      break;
  }
}

$("nudge-th").addEventListener("input", (e) => setNudgeThreshold(+e.target.value));
$("mic").addEventListener("click", () => (connected ? stop() : start()));
buildDims();
setNudgeThreshold(NUDGE_TH.default);
