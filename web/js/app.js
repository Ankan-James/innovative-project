/**
 * SLMCI — Application Entry Point
 * Orchestrates UI interactions, artwork rendering, and controller wiring
 */

document.addEventListener("DOMContentLoaded", () => {
  // =========================================================================
  // 1. Initialize Overlay Canvas
  // =========================================================================

  const overlayCanvas = document.getElementById("overlay");
  const overlay = new HandOverlay(overlayCanvas);

  // =========================================================================
  // 3. Initialize API Client
  // =========================================================================

  const apiClient = new ApiClient("/predict-video");

  // =========================================================================
  // 4. Initialize Interpreter UI
  // =========================================================================

  const interpreterUI = new InterpreterUI({
    letterEl: document.getElementById("recognizedSign"),
    scoreEl: document.getElementById("modelScore"),
    statusEl: document.getElementById("dynamicStatus"),
    votingGridEl: document.getElementById("votingGrid"),
    voteFractionEl: document.getElementById("voteFraction"),
    formedTextEl: document.getElementById("formedText"),
    autocorrectBadgeEl: document.getElementById("autocorrectBadge")
  });

  // =========================================================================
  // 5. Initialize Camera Controller
  // =========================================================================

  const cameraVideo = document.getElementById("camera");
  const cameraPlaceholder = document.getElementById("cameraPlaceholder");
  const cameraStatus = document.getElementById("cameraStatus");
  const startButton = document.getElementById("startButton");
  const stopButton = document.getElementById("stopButton");
  const nameModeCheckbox = document.getElementById("nameMode");
  const clearTextBtn = document.getElementById("clearTextBtn");

  const cameraController = new CameraController({
    videoElement: cameraVideo,
    placeholderElement: cameraPlaceholder,
    statusElement: cameraStatus,
    startButton: startButton,
    stopButton: stopButton,
    nameModeCheckbox: nameModeCheckbox,
    apiClient: apiClient,
    interpreterUI: interpreterUI,
    overlay: overlay
  });

  // =========================================================================
  // 6. Initialize Text-to-Speech (TTS)
  // =========================================================================

  const ttsSpeakBtn = document.getElementById("ttsSpeakBtn");
  const ttsWaveform = document.getElementById("ttsWaveform");
  const formedTextEl = document.getElementById("formedText");

  const tts = new TextToSpeech({
    speakBtn: ttsSpeakBtn,
    waveformEl: ttsWaveform,
    formedTextEl: formedTextEl
  });

  interpreterUI.onTextChange = () => {
    tts.updateButtonState();
  };

  // =========================================================================
  // 7. UI Event Bindings
  // =========================================================================

  if (startButton) {
    startButton.addEventListener("click", () => cameraController.start());
  }

  if (stopButton) {
    stopButton.addEventListener("click", () => cameraController.stop());
  }

  if (nameModeCheckbox) {
    nameModeCheckbox.addEventListener("change", (e) => {
      interpreterUI.setNameMode(e.target.checked);
    });
  }

  if (clearTextBtn) {
    clearTextBtn.addEventListener("click", () => {
      cameraController.resetSession();
      if (tts) {
        tts.stop();
        tts.updateButtonState();
      }
    });
  }

  // Navigation and Explore smooth scroll
  document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener("click", function (e) {
      const targetId = this.getAttribute("href");
      if (targetId && targetId !== "#") {
        const targetElement = document.querySelector(targetId);
        if (targetElement) {
          e.preventDefault();
          targetElement.scrollIntoView({ behavior: "smooth" });
        }
      }
    });
  });
});
