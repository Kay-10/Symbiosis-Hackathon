const state = {
  sessionId: null,
  messages: [],
  recording: false,
  processing: false,
  mediaRecorder: null,
  audioChunks: [],
  audioContext: null,
  analyser: null,
  silenceStart: null,
  silenceThreshold: 0.01,
  silenceDuration: 1500,
  analyserData: null,
  stream: null,
  currentController: null,
  speakingUtterance: null,
};

const elements = {
  status: document.getElementById("status-pill"),
  messageList: document.getElementById("message-list"),
  textInput: document.getElementById("text-input"),
  sendBtn: document.getElementById("send-btn"),
  talkBtn: document.getElementById("talk-btn"),
  stopBtn: document.getElementById("stop-btn"),
  messageTemplate: document.getElementById("message-template"),
  chatWindow: document.getElementById("chat-window"),
};

async function init() {
  await ensureSession();
  await loadHistory();
  bindEvents();
}

async function ensureSession() {
  let sessionId = localStorage.getItem("sakhiSessionId");
  if (!sessionId) {
    const res = await fetch("/api/session", { method: "POST" });
    const data = await res.json();
    sessionId = data.session_id;
    localStorage.setItem("sakhiSessionId", sessionId);
  }
  state.sessionId = sessionId;
}

async function loadHistory() {
  const res = await fetch(`/api/history?session_id=${state.sessionId}`);
  if (!res.ok) return;
  const data = await res.json();
  state.messages = data.history || [];
  renderMessages();
}

function bindEvents() {
  elements.sendBtn.addEventListener("click", () => {
    const text = elements.textInput.value.trim();
    if (!text) return;
    elements.textInput.value = "";
    handleUserMessage(text, "text");
  });

  elements.textInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      elements.sendBtn.click();
    }
  });

  elements.talkBtn.addEventListener("click", startRecording);
  elements.stopBtn.addEventListener("click", stopInteraction);
}

function renderMessages() {
  elements.messageList.innerHTML = "";
  state.messages.forEach((entry) => appendMessage(entry.role, entry.content));
  scrollToBottom();
}

function appendMessage(role, content) {
  const li = elements.messageTemplate.content.cloneNode(true);
  const root = li.querySelector(".message");
  root.classList.toggle("user", role === "user");
  root.classList.toggle("assistant", role === "assistant");
  root.querySelector(".avatar").textContent = role === "assistant" ? "🌸" : "👩";
  root.querySelector(".meta").textContent = role === "assistant" ? "Sakhi" : "You";
  root.querySelector(".text").innerText = content;
  elements.messageList.appendChild(li);
  scrollToBottom();
}

function scrollToBottom() {
  elements.chatWindow.scrollTo({ top: elements.chatWindow.scrollHeight, behavior: "smooth" });
}

function setStatus(label, variant = "ready") {
  elements.status.textContent = label;
  elements.status.classList.toggle("busy", variant === "busy");
}

function setControls({ recording = state.recording, processing = state.processing }) {
  state.recording = recording;
  state.processing = processing;

  elements.talkBtn.disabled = recording || processing;
  elements.sendBtn.disabled = processing;
  elements.textInput.disabled = processing;
  elements.stopBtn.disabled = !recording && !processing;

  if (recording) {
    setStatus("Listening…", "busy");
  } else if (processing) {
    setStatus("Working…", "busy");
  } else {
    setStatus("Ready");
  }
}

async function handleUserMessage(message, mode) {
  stopSpeech();
  state.messages.push({ role: "user", content: message });
  appendMessage("user", message);
  setControls({ processing: true, recording: false });

  try {
    const payload = { session_id: state.sessionId, message };
    const controller = new AbortController();
    state.currentController = controller;

    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal: controller.signal,
    });

    if (!res.ok) {
      const detail = (await res.json()).detail || "Unable to reach Sakhi";
      state.messages.push({ role: "assistant", content: detail });
      appendMessage("assistant", detail);
      return;
    }

    const data = await res.json();
    state.messages = data.history || [];
    renderMessages();
    speakText(data.assistant_message, data.language);
  } catch (error) {
    if (error.name !== "AbortError") {
      appendMessage("assistant", "मुझे कनेक्शन में परेशानी हो रही है, कृपया थोड़ी देर बाद प्रयास करें।");
      state.messages.push({
        role: "assistant",
        content: "मुझे कनेक्शन में परेशानी हो रही है, कृपया थोड़ी देर बाद प्रयास करें।",
      });
    }
  } finally {
    state.currentController = null;
    if (!state.speakingUtterance) {
      setControls({ processing: false, recording: false });
    }
  }
}

async function startRecording() {
  stopSpeech();
  if (state.recording || state.processing) return;
  try {
    state.stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (error) {
    alert("Microphone access is required to talk to Sakhi.");
    return;
  }

  state.audioContext = new AudioContext();
  const source = state.audioContext.createMediaStreamSource(state.stream);
  state.analyser = state.audioContext.createAnalyser();
  state.analyser.fftSize = 2048;
  state.analyserData = new Float32Array(state.analyser.fftSize);
  source.connect(state.analyser);

  state.audioChunks = [];
  state.mediaRecorder = new MediaRecorder(state.stream, { mimeType: "audio/webm" });
  state.mediaRecorder.ondataavailable = (event) => {
    if (event.data.size > 0) {
      state.audioChunks.push(event.data);
    }
  };
  state.mediaRecorder.onstop = handleRecordingStop;
  state.mediaRecorder.start(250);

  setControls({ recording: true, processing: false });
  monitorSilence(performance.now());
}

function monitorSilence(lastTimestamp) {
  if (!state.recording || !state.analyser) return;
  state.analyser.getFloatTimeDomainData(state.analyserData);
  let sum = 0;
  for (let i = 0; i < state.analyserData.length; i += 1) {
    const sample = state.analyserData[i];
    sum += sample * sample;
  }
  const rms = Math.sqrt(sum / state.analyserData.length);
  const now = performance.now();
  if (rms < state.silenceThreshold) {
    if (!state.silenceStart) {
      state.silenceStart = now;
    } else if (now - state.silenceStart > state.silenceDuration) {
      stopRecording();
      return;
    }
  } else {
    state.silenceStart = null;
  }
  requestAnimationFrame(monitorSilence);
}

function stopRecording() {
  if (!state.recording) return;
  state.recording = false;
  if (state.mediaRecorder && state.mediaRecorder.state !== "inactive") {
    state.mediaRecorder.stop();
  }
  cleanupStream();
  setStatus("Processing…", "busy");
}

function cleanupStream() {
  if (state.stream) {
    state.stream.getTracks().forEach((track) => track.stop());
    state.stream = null;
  }
  if (state.audioContext) {
    state.audioContext.close();
    state.audioContext = null;
  }
  state.analyser = null;
  state.analyserData = null;
  state.silenceStart = null;
}

async function handleRecordingStop() {
  cleanupStream();
  if (!state.audioChunks.length) {
    setControls({ processing: false, recording: false });
    return;
  }
  setControls({ recording: false, processing: true });

  try {
    const webmBlob = new Blob(state.audioChunks, { type: "audio/webm" });
    const wavBlob = await convertWebmToWav(webmBlob);
    await transcribeAndSend(wavBlob);
  } catch (error) {
    console.error(error);
    appendMessage("assistant", "मैं आपकी आवाज़ समझ नहीं पाई, कृपया फिर से बताएँ।");
    state.messages.push({
      role: "assistant",
      content: "मैं आपकी आवाज़ समझ नहीं पाई, कृपया फिर से बताएँ।",
    });
  } finally {
    state.audioChunks = [];
  }
}

async function convertWebmToWav(blob) {
  const arrayBuffer = await blob.arrayBuffer();
  const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
  const wavBuffer = audioBufferToWav(audioBuffer);
  await audioCtx.close();
  return new Blob([wavBuffer], { type: "audio/wav" });
}

function audioBufferToWav(buffer) {
  const numChannels = buffer.numberOfChannels;
  const sampleRate = buffer.sampleRate;
  const format = 1;
  const bitDepth = 16;

  const samples = interleave(buffer);
  const blockAlign = numChannels * (bitDepth / 8);
  const byteRate = sampleRate * blockAlign;
  const dataSize = samples.length * (bitDepth / 8);
  const bufferLength = 44 + dataSize;
  const view = new DataView(new ArrayBuffer(bufferLength));
  let offset = 0;

  function writeString(str) {
    for (let i = 0; i < str.length; i += 1) {
      view.setUint8(offset + i, str.charCodeAt(i));
    }
    offset += str.length;
  }

  function writeUint16(data) {
    view.setUint16(offset, data, true);
    offset += 2;
  }

  function writeUint32(data) {
    view.setUint32(offset, data, true);
    offset += 4;
  }

  writeString("RIFF");
  writeUint32(36 + dataSize);
  writeString("WAVE");
  writeString("fmt ");
  writeUint32(16);
  writeUint16(format);
  writeUint16(numChannels);
  writeUint32(sampleRate);
  writeUint32(byteRate);
  writeUint16(blockAlign);
  writeUint16(bitDepth);
  writeString("data");
  writeUint32(dataSize);

  const volume = Math.pow(2, bitDepth - 1) - 1;
  for (let i = 0; i < samples.length; i += 1, offset += 2) {
    const sample = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, sample * volume, true);
  }

  return view.buffer;
}

function interleave(buffer) {
  if (buffer.numberOfChannels === 1) {
    return buffer.getChannelData(0);
  }
  const length = buffer.length * buffer.numberOfChannels;
  const result = new Float32Array(length);
  const input = [];
  for (let channel = 0; channel < buffer.numberOfChannels; channel += 1) {
    input.push(buffer.getChannelData(channel));
  }
  let index = 0;
  for (let i = 0; i < buffer.length; i += 1) {
    for (let channel = 0; channel < buffer.numberOfChannels; channel += 1) {
      result[index] = input[channel][i];
      index += 1;
    }
  }
  return result;
}

async function transcribeAndSend(blob) {
  const formData = new FormData();
  formData.append("session_id", state.sessionId);
  formData.append("audio", blob, "speech.wav");

  const controller = new AbortController();
  state.currentController = controller;

  setStatus("Understanding your voice…", "busy");
  try {
    const res = await fetch("/api/transcribe", {
      method: "POST",
      body: formData,
      signal: controller.signal,
    });
    if (!res.ok) {
      throw new Error("transcribe failed");
    }
    const data = await res.json();
    const text = data.text.trim();
    if (!text) {
      throw new Error("empty transcription");
    }
    handleUserMessage(text, "voice");
  } catch (error) {
    if (error.name !== "AbortError") {
      appendMessage("assistant", "मुझे आपकी आवाज़ समझ नहीं पाई, कृपया फिर से बताएँ।");
      state.messages.push({
        role: "assistant",
        content: "मुझे आपकी आवाज़ समझ नहीं पाई, कृपया फिर से बताएँ।",
      });
      setControls({ processing: false, recording: false });
    }
  } finally {
    state.currentController = null;
  }
}

function stopInteraction() {
  stopSpeech();
  if (state.recording) {
    stopRecording();
    return;
  }
  if (state.currentController) {
    state.currentController.abort();
    state.currentController = null;
  }
  setControls({ recording: false, processing: false });
}

function stopSpeech() {
  if (state.speakingUtterance) {
    window.speechSynthesis.cancel();
    state.speakingUtterance = null;
  }
}

function speakText(text, language) {
  if (!("speechSynthesis" in window)) {
    return;
  }
  stopSpeech();
  const utterance = new SpeechSynthesisUtterance(text);
  if (language) {
    utterance.lang = language;
  }
  utterance.rate = 0.95;
  utterance.pitch = 1;
  utterance.onstart = () => {
    setControls({ processing: true, recording: false });
    setStatus("Speaking…", "busy");
  };
  utterance.onend = () => {
    state.speakingUtterance = null;
    setControls({ processing: false, recording: false });
  };
  state.speakingUtterance = utterance;
  window.speechSynthesis.speak(utterance);
}

init();
