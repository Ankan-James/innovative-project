/**
 * SLMCI — Text-to-Speech (TTS) & Waveform Controller
 * Browser-native speech synthesis using the Web Speech API
 * (window.speechSynthesis & SpeechSynthesisUtterance)
 *
 * Lightweight, zero-dependency, and optimized for low-spec systems.
 */

class TextToSpeech {
  constructor(options = {}) {
    this.speakBtn = options.speakBtn || null;
    this.waveformEl = options.waveformEl || null;
    this.formedTextEl = options.formedTextEl || null;
    this.placeholderText = options.placeholderText || "Your interpreted text will appear here...";

    this.synth = typeof window !== "undefined" && "speechSynthesis" in window ? window.speechSynthesis : null;
    this.isSupported = !!(this.synth && typeof window.SpeechSynthesisUtterance !== "undefined");

    this.state = "idle"; // 'idle' | 'speaking' | 'stopping' | 'error'
    this.utteranceCounter = 0;
    this.currentUtterance = null;
    this.selectedVoice = null;
    this.observer = null;

    this.init();
  }

  /**
   * Initializes voice list, event listeners, and formed text observer
   */
  init() {
    if (!this.isSupported) {
      if (this.speakBtn) {
        this.speakBtn.disabled = true;
        this.speakBtn.setAttribute("aria-disabled", "true");
        this.speakBtn.title = "Text-to-speech is not supported in this browser";
      }
      return;
    }

    // Load available voices
    this.loadVoices();
    if (this.synth.onvoiceschanged !== undefined) {
      this.synth.onvoiceschanged = () => this.loadVoices();
    }

    // Bind Speak/Stop button click
    if (this.speakBtn) {
      this.speakBtn.addEventListener("click", () => this.toggleSpeech());
      this.speakBtn.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          this.toggleSpeech();
        }
      });
    }

    // Observe formedTextEl for real-time changes
    if (this.formedTextEl) {
      this.observer = new MutationObserver(() => {
        this.updateButtonState();
      });

      this.observer.observe(this.formedTextEl, {
        childList: true,
        characterData: true,
        subtree: true,
        attributes: true,
        attributeFilter: ["class"]
      });
    }

    // Initial state check
    this.updateButtonState();
  }

  /**
   * Loads browser voices and selects the most natural English voice available
   */
  loadVoices() {
    if (!this.synth) return;
    const voices = this.synth.getVoices();
    if (!voices || voices.length === 0) return;

    // Preference: 1. Natural/Neural/Google en-US -> 2. Standard en-US -> 3. Any English voice -> 4. Default voice
    const isEnUS = v => v.lang === "en-US" || v.lang === "en_US";
    const isEn = v => v.lang && v.lang.toLowerCase().startsWith("en");
    const isNatural = v => /natural|neural|google|samantha|zira|david/i.test(v.name);

    const naturalEnUS = voices.find(v => isEnUS(v) && isNatural(v));
    const anyEnUS = voices.find(v => isEnUS(v));
    const naturalEn = voices.find(v => isEn(v) && isNatural(v));
    const anyEn = voices.find(v => isEn(v));
    const defaultVoice = voices.find(v => v.default);

    this.selectedVoice = naturalEnUS || anyEnUS || naturalEn || anyEn || defaultVoice || voices[0] || null;
  }

  /**
   * Retrieves clean, actual text from the formed-text display
   * Returns empty string if empty or placeholder
   */
  getFormedText() {
    if (!this.formedTextEl) return "";
    if (this.formedTextEl.classList.contains("empty")) return "";

    const text = this.formedTextEl.textContent || "";
    const trimmed = text.trim();

    if (!trimmed || trimmed === this.placeholderText) {
      return "";
    }

    return trimmed;
  }

  /**
   * Toggles between starting and stopping speech
   */
  toggleSpeech() {
    if (this.state === "speaking") {
      this.stop();
    } else {
      this.speak();
    }
  }

  /**
   * Converts formed text to speech using SpeechSynthesisUtterance
   */
  speak() {
    if (!this.isSupported) return;

    const text = this.getFormedText();
    if (!text) {
      this.setIdleState();
      this.updateButtonState();
      return;
    }

    // Cancel any previous/lingering speech
    this.synth.cancel();

    // Create fresh utterance
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = "en-US";
    utterance.rate = 1.0;
    utterance.pitch = 1.0;
    utterance.volume = 1.0;

    if (this.selectedVoice) {
      utterance.voice = this.selectedVoice;
    }

    const currentId = ++this.utteranceCounter;

    utterance.onstart = () => {
      if (this.utteranceCounter !== currentId) return;
      this.setSpeakingState();
    };

    utterance.onend = () => {
      if (this.utteranceCounter !== currentId) return;
      this.setIdleState();
    };

    utterance.onerror = (event) => {
      // Ignore 'canceled' error produced by intentional stop()
      if (event.error === "canceled" || event.error === "interrupted") {
        return;
      }
      if (this.utteranceCounter !== currentId) return;
      this.handleError(event.error || "Speech synthesis failed");
    };

    this.currentUtterance = utterance;

    // Immediately reflect active intent in UI for responsive feedback
    this.setSpeakingState();

    // Speak through browser engine
    try {
      this.synth.speak(utterance);

      // Edge case: resume suspended audio context in Chromium if needed
      if (this.synth.paused) {
        this.synth.resume();
      }
    } catch (err) {
      this.handleError(err.message);
    }
  }

  /**
   * Stops active speech playback immediately
   */
  stop() {
    this.state = "stopping";
    if (this.synth) {
      try {
        this.synth.cancel();
      } catch (_) {}
    }
    this.setIdleState();
  }

  /**
   * Sets UI into active speaking state
   */
  setSpeakingState() {
    this.state = "speaking";

    if (this.speakBtn) {
      this.speakBtn.classList.add("is-speaking");
      this.speakBtn.setAttribute("aria-label", "Stop speech");
      this.speakBtn.setAttribute("aria-pressed", "true");
      this.speakBtn.title = "Stop speech";
      this.speakBtn.disabled = false;
      this.speakBtn.removeAttribute("aria-disabled");
    }

    if (this.waveformEl) {
      this.waveformEl.classList.add("is-speaking");
    }
  }

  /**
   * Restores UI to idle state
   */
  setIdleState() {
    this.state = "idle";
    this.currentUtterance = null;

    if (this.speakBtn) {
      this.speakBtn.classList.remove("is-speaking");
      this.speakBtn.setAttribute("aria-label", "Speak formed text");
      this.speakBtn.setAttribute("aria-pressed", "false");
      this.speakBtn.title = "Speak formed text";
    }

    if (this.waveformEl) {
      this.waveformEl.classList.remove("is-speaking");
    }

    this.updateButtonState();
  }

  /**
   * Non-intrusive error handler
   */
  handleError(errorMessage) {
    this.state = "error";
    this.setIdleState();
    console.warn("SLMCI TTS Error:", errorMessage);
  }

  /**
   * Enables or disables the Speak button based on text availability
   */
  updateButtonState() {
    if (!this.speakBtn) return;

    if (!this.isSupported) {
      this.speakBtn.disabled = true;
      this.speakBtn.setAttribute("aria-disabled", "true");
      this.speakBtn.title = "Text-to-speech not supported";
      return;
    }

    // While speaking, button should remain enabled so user can click Stop
    if (this.state === "speaking") {
      this.speakBtn.disabled = false;
      this.speakBtn.removeAttribute("aria-disabled");
      return;
    }

    const hasText = this.getFormedText().length > 0;
    this.speakBtn.disabled = !hasText;

    if (!hasText) {
      this.speakBtn.setAttribute("aria-disabled", "true");
      this.speakBtn.title = "No text to speak";
    } else {
      this.speakBtn.removeAttribute("aria-disabled");
      this.speakBtn.title = "Speak formed text";
    }
  }

  /**
   * Cleanup observer and cancel synthesis
   */
  destroy() {
    if (this.observer) {
      this.observer.disconnect();
      this.observer = null;
    }
    if (this.synth) {
      this.synth.cancel();
    }
  }
}

if (typeof window !== "undefined") {
  window.TextToSpeech = TextToSpeech;
}
