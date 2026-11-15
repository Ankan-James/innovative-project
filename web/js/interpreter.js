/**
 * SLMCI — Interpreter UI Component
 * Manages recognition display, dynamic status states, 4x2 voting grid, and formed text
 */

class InterpreterUI {
  constructor(elements) {
    this.letterEl = elements.letterEl;
    this.scoreEl = elements.scoreEl;
    this.statusEl = elements.statusEl;
    this.votingGridEl = elements.votingGridEl;
    this.voteFractionEl = elements.voteFractionEl;
    this.formedTextEl = elements.formedTextEl;
    this.autocorrectBadgeEl = elements.autocorrectBadgeEl;
    
    this.totalVoteCells = 8;
    this.initVotingGrid();
    this.reset();
  }

  /**
   * Generates the 4 rows x 2 columns = 8 squares in the DOM
   */
  initVotingGrid() {
    if (!this.votingGridEl) return;
    this.votingGridEl.innerHTML = "";
    this.cells = [];

    for (let i = 0; i < this.totalVoteCells; i++) {
      const cell = document.createElement("div");
      cell.className = "vote-cell neutral";
      cell.setAttribute("data-index", i);
      this.votingGridEl.appendChild(cell);
      this.cells.push(cell);
    }
  }

  /**
   * Resets the entire detector UI to inactive camera state
   */
  reset() {
    if (this.letterEl) this.letterEl.textContent = "—";
    if (this.scoreEl) this.scoreEl.textContent = "—";
    if (this.statusEl) {
      this.statusEl.textContent = "Camera inactive";
      this.statusEl.className = "recognition-status-text";
    }
    this.setVotingGridNeutral();
    if (this.voteFractionEl) this.voteFractionEl.textContent = "0/8";
  }

  /**
   * Sets status for camera active, waiting for hand
   */
  setWaitingForHand() {
    if (this.letterEl) this.letterEl.textContent = "—";
    if (this.scoreEl) this.scoreEl.textContent = "—";
    if (this.statusEl) {
      this.statusEl.textContent = "Waiting for hand detection...";
      this.statusEl.className = "recognition-status-text";
    }
    this.setVotingGridNeutral();
    if (this.voteFractionEl) this.voteFractionEl.textContent = "0/8";
  }

  /**
   * Sets status when hand is detected and currently analyzing
   */
  setAnalyzing() {
    if (this.statusEl) {
      this.statusEl.textContent = "Hand detected — Analyzing sign...";
      this.statusEl.className = "recognition-status-text";
    }
  }

  /**
   * Updates UI from backend prediction result
   *
   * @param {Object} result - JSON response from /predict-video
   */
  displayResult(result) {
    if (!result) return;

    // Update formed text (accumulated letters/words)
    if (result.text && result.text.trim().length > 0) {
      this.formedTextEl.textContent = result.text;
      this.formedTextEl.classList.remove("empty");
    } else {
      this.formedTextEl.textContent = "Your interpreted text will appear here...";
      this.formedTextEl.classList.add("empty");
    }

    if (typeof this.onTextChange === "function") {
      this.onTextChange(result.text || "");
    }

    const framesAnalyzed = result.frames_analyzed || 8;
    const votes = typeof result.votes === "number" ? result.votes : 0;

    // State 1: No hand detected
    if (!result.hand_detected) {
      if (this.letterEl) this.letterEl.textContent = "—";
      if (this.scoreEl) this.scoreEl.textContent = "—";
      if (this.statusEl) {
        this.statusEl.textContent = "No hand detected";
        this.statusEl.className = "recognition-status-text";
      }
      this.setVotingGridNeutral();
      if (this.voteFractionEl) this.voteFractionEl.textContent = "0/8";
      return;
    }

    // State 2: Hand detected, but predictions unstable (insufficient matching votes)
    if (!result.label) {
      if (this.letterEl) this.letterEl.textContent = "—";
      if (this.scoreEl) this.scoreEl.textContent = "—";
      if (this.statusEl) {
        this.statusEl.textContent = "Hand detected — Hold your sign steady";
        this.statusEl.className = "recognition-status-text status-unstable";
      }
      this.setVotingGridNeutral();
      if (this.voteFractionEl) this.voteFractionEl.textContent = `${votes}/${framesAnalyzed}`;
      return;
    }

    // State 3: Successful recognition
    if (this.letterEl) this.letterEl.textContent = result.label;
    if (this.scoreEl) {
      const conf = typeof result.confidence === "number" ? result.confidence.toFixed(1) : result.confidence;
      this.scoreEl.textContent = `${conf}%`;
    }
    if (this.statusEl) {
      this.statusEl.textContent = "Hand detected — Sign recognized successfully";
      this.statusEl.className = "recognition-status-text status-success";
    }

    // Update 4x2 voting grid aggregate representation
    this.updateVotingGrid(votes, framesAnalyzed);
  }

  /**
   * Fills the 4x2 voting grid:
   * First `votes` cells green (matching), remaining up to `frames` red (nonmatching),
   * any leftover neutral.
   */
  updateVotingGrid(votes, frames = 8) {
    if (!this.cells || this.cells.length === 0) return;

    for (let i = 0; i < this.cells.length; i++) {
      const cell = this.cells[i];
      if (i < votes) {
        cell.className = "vote-cell matching";
      } else if (i < frames) {
        cell.className = "vote-cell nonmatching";
      } else {
        cell.className = "vote-cell neutral";
      }
    }

    if (this.voteFractionEl) {
      this.voteFractionEl.textContent = `${votes}/${frames}`;
    }
  }

  /**
   * Resets all cells to neutral gray
   */
  setVotingGridNeutral() {
    if (!this.cells) return;
    this.cells.forEach(cell => {
      cell.className = "vote-cell neutral";
    });
  }

  /**
   * Updates Name Mode badge
   */
  setNameMode(isNameMode) {
    if (!this.autocorrectBadgeEl) return;
    if (isNameMode) {
      this.autocorrectBadgeEl.textContent = "Name Mode active";
      this.autocorrectBadgeEl.classList.add("badge-name-mode");
    } else {
      this.autocorrectBadgeEl.textContent = "Autocorrect enabled";
      this.autocorrectBadgeEl.classList.remove("badge-name-mode");
    }
  }

  /**
   * Clears formed text
   */
  clearFormedText() {
    if (!this.formedTextEl) return;
    this.formedTextEl.textContent = "Your interpreted text will appear here...";
    this.formedTextEl.classList.add("empty");

    if (typeof this.onTextChange === "function") {
      this.onTextChange("");
    }
  }
}

if (typeof window !== "undefined") {
  window.InterpreterUI = InterpreterUI;
}
