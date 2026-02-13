(function () {
  "use strict";

  const state = {
    activeTab: "theory",
    docs: ["alex", "james", "mary", "anna", "john", "emily", "luke", "olivia", "noah", "sophia"],
    isTraining: false,
    traceEnabled: false,
    trainProgress: [],
    recentPredictions: [],
    paramCount: 0,
    isInitialized: false,
    explanationByTopic: {
      "Autograd": "What it is:\nAutograd is an automatic way to compute derivatives. Every math operation creates a node that stores both a value (Data) and how sensitive loss is to that value (Grad).\n\nHow it works here:\n1) Forward pass: the model computes logits and loss.\n2) Backward pass: gradients are propagated from loss back through all connected nodes using the chain rule.\n3) Update: Adam uses those gradients to nudge parameters in directions that should reduce future loss.\n\nWhy it matters:\nWithout autograd, each derivative would need to be hand-coded and easy to break. With autograd, the model can safely combine many operations and still learn from errors.",
      "Transformer": "What it is:\nA Transformer is a next-token predictor built from repeated blocks. Each block mixes information across positions (attention) and then transforms features (MLP).\n\nHow it works in this app:\n1) Character token + position embedding are combined.\n2) RMSNorm stabilizes magnitudes.\n3) Attention reads prior context using query/key/value projections.\n4) Residual connection keeps useful prior signal.\n5) MLP expands then compresses features for richer representation.\n6) Final linear head converts features to logits over possible next characters.\n\nWhy it matters:\nThis structure can model dependencies between distant characters while remaining efficient enough for iterative training and generation in a browser-connected demo.",
      "Attention": "What it is:\nAttention is a weighted lookup over previous tokens. At each step, the model asks: 'Which earlier positions are most relevant to predicting the next character?'\n\nHow the score is formed:\n- Query (current state) is compared with Keys (past states).\n- Dot products are scaled, then softmax converts them into probabilities.\n- Those probabilities weight the Values (past content) to build a context vector.\n\nInterpretation:\nHigher attention weight means that prior token had more influence on this step. In trace output, top candidates reflect the combined effect of these context-aware features plus output projection.",
      "Go Backend": "What it is:\nThe Go backend is the execution engine for model state, math, and APIs. The UI only visualizes data; learning and sampling happen in Go.\n\nRuntime responsibilities:\n- /api/init: build model parameters from docs + config.\n- /api/train: run one training step and return metrics for chart/feedback panel.\n- /api/generate: sample next-token sequence from current model state.\n- /api/generate_trace: return step-by-step candidate probabilities and selection reasons.\n\nWhy Go helps here:\nGo gives predictable performance, simple concurrency control via mutexes, and easy deployment as a single binary serving both model APIs and static UI assets."
    }
  };

  const el = {
    menuTheory: document.getElementById("menu-theory"),
    menuTrain: document.getElementById("menu-train"),
    menuInference: document.getElementById("menu-inference"),
    tabTheory: document.getElementById("tab-theory"),
    tabTrain: document.getElementById("tab-train"),
    tabInference: document.getElementById("tab-inference"),
    viewTheory: document.getElementById("view-theory"),
    viewTrain: document.getElementById("view-train"),
    viewInference: document.getElementById("view-inference"),
    explanationBox: document.getElementById("explanationBox"),
    explanationText: document.getElementById("explanationText"),
    startTrainBtn: document.getElementById("startTrainBtn"),
    stopTrainBtn: document.getElementById("stopTrainBtn"),
    generateBtn: document.getElementById("generateBtn"),
    traceBtn: document.getElementById("traceBtn"),
    generatedText: document.getElementById("generatedText"),
    traceWindow: document.getElementById("traceWindow"),
    traceSummary: document.getElementById("traceSummary"),
    traceList: document.getElementById("traceList"),
    lossCanvas: document.getElementById("lossCanvas"),
    lossLabel: document.getElementById("lossLabel"),
    paramCount: document.getElementById("paramCount"),
    contextChar: document.getElementById("contextChar"),
    targetChar: document.getElementById("targetChar"),
    predictedChar: document.getElementById("predictedChar"),
    targetProb: document.getElementById("targetProb"),
    predictedProb: document.getElementById("predictedProb"),
    tokenTape: document.getElementById("tokenTape"),
    docsList: document.getElementById("docsList"),
    newDocInput: document.getElementById("newDocInput"),
    addDocBtn: document.getElementById("addDocBtn")
  };

  function updateMenuTabState(tab) {
    if (el.menuTheory) {
      el.menuTheory.classList.toggle("active", tab === "theory");
    }
    if (el.menuTrain) {
      el.menuTrain.classList.toggle("active", tab === "train");
    }
    if (el.menuInference) {
      el.menuInference.classList.toggle("active", tab === "inference");
    }
  }

  function setTraceEnabled(enabled) {
    state.traceEnabled = enabled;
    el.traceBtn.classList.toggle("active", enabled);
    el.traceBtn.textContent = enabled ? "Explain Choice: On" : "Explain Choice: Off";
    if (enabled) {
      el.traceWindow.classList.remove("hidden-down");
    } else {
      el.traceWindow.classList.add("hidden-down");
    }
  }

  function getConfidenceColor(value) {
    if (value < 0.65) {
      return "confidence-low";
    }
    if (value < 0.9) {
      return "confidence-mid";
    }
    return "confidence-high";
  }

  function setTargetConfidenceClass(value) {
    const cls = getConfidenceColor(value);
    el.targetProb.classList.remove("confidence-high", "confidence-mid", "confidence-low");
    el.targetProb.classList.add(cls);
  }

  function setActiveTab(tab) {
    state.activeTab = tab;
    el.tabTheory.classList.toggle("active", tab === "theory");
    el.tabTrain.classList.toggle("active", tab === "train");
    el.tabInference.classList.toggle("active", tab === "inference");
    el.viewTheory.classList.toggle("hidden", tab !== "theory");
    el.viewTrain.classList.toggle("hidden", tab !== "train");
    el.viewInference.classList.toggle("hidden", tab !== "inference");
    updateMenuTabState(tab);
  }

  async function initModel() {
    const config = {
      n_embd: 16,
      n_head: 4,
      n_layer: 1,
      block_size: 16,
      learning_rate: 0.05
    };
    const res = await fetch("/api/init", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ docs: state.docs, config: config })
    });
    if (!res.ok) {
      throw new Error("failed to initialize model");
    }
    const data = await res.json();
    state.paramCount = data.params || 0;
    state.isInitialized = true;
    state.trainProgress = [];
    state.recentPredictions = [];
    el.paramCount.textContent = String(state.paramCount);
    el.lossLabel.textContent = "Current Loss: N/A";
    el.contextChar.textContent = "N/A";
    el.targetChar.textContent = "N/A";
    el.predictedChar.textContent = "N/A";
    el.targetProb.textContent = "0.0000";
    el.predictedProb.textContent = "0.0000";
    setTargetConfidenceClass(1);
    el.tokenTape.textContent = "-";
    drawChart();
  }

  function renderDocs() {
    el.docsList.innerHTML = "";
    state.docs.forEach(function (doc, index) {
      const row = document.createElement("div");
      row.className = "flex justify-between items-center p-1 border-b border-black text-xs hover:bg-black hover:text-white group";

      const text = document.createElement("span");
      text.className = "code-font";
      text.textContent = doc;

      const remove = document.createElement("button");
      remove.className = "cursor-pointer font-bold px-1";
      remove.textContent = "x";
      remove.addEventListener("click", function () {
        state.docs.splice(index, 1);
        renderDocs();
        initModel().catch(console.error);
      });

      row.appendChild(text);
      row.appendChild(remove);
      el.docsList.appendChild(row);
    });
  }

  function drawChart() {
    const canvas = el.lossCanvas;
    if (!canvas) {
      return;
    }
    const ctx = canvas.getContext("2d");
    if (!ctx) {
      return;
    }

    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    if (state.trainProgress.length < 2) {
      return;
    }

    const maxLoss = Math.max.apply(
      null,
      state.trainProgress.map(function (p) {
        return p.loss;
      }).concat([5])
    );

    // Keep trend line neutral for readability.
    ctx.strokeStyle = "#111111";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    state.trainProgress.forEach(function (p, i) {
      const x = (i / 49) * w;
      const y = h - (p.loss / maxLoss) * h;
      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });
    ctx.stroke();

    // Overlay confidence as subtle dots.
    state.trainProgress.forEach(function (p, i) {
      const x = (i / 49) * w;
      const y = h - (p.loss / maxLoss) * h;
      const prob = Number(p.targetProb || 0);
      let color = "#b00020";
      if (prob >= 0.9) {
        color = "#0a7a0a";
      } else if (prob >= 0.65) {
        color = "#b58900";
      }
      ctx.fillStyle = color;
      ctx.strokeStyle = "#000000";
      ctx.lineWidth = 0.6;
      ctx.beginPath();
      ctx.arc(x, y, 2.2, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
    });
  }

  async function startTraining() {
    if (!state.isInitialized || state.isTraining) {
      return;
    }
    state.isTraining = true;
    el.startTrainBtn.classList.add("hidden");
    el.stopTrainBtn.classList.remove("hidden");

    while (state.isTraining) {
      try {
        const res = await fetch("/api/train", { method: "POST" });
        if (!res.ok) {
          throw new Error("training request failed");
        }
        const data = await res.json();
        state.trainProgress.push({ step: data.step, loss: data.loss, targetProb: Number(data.target_prob || 0) });
        if (state.trainProgress.length > 50) {
          state.trainProgress.shift();
        }
        el.contextChar.textContent = data.context_char || "N/A";
        el.targetChar.textContent = data.target_char || "N/A";
        el.predictedChar.textContent = data.predicted_char || "N/A";
        const targetProb = Number(data.target_prob || 0);
        el.targetProb.textContent = targetProb.toFixed(4);
        setTargetConfidenceClass(targetProb);
        el.predictedProb.textContent = Number(data.predicted_prob || 0).toFixed(4);
        if (!state.recentPredictions) {
          state.recentPredictions = [];
        }
        state.recentPredictions.push(data.predicted_char || "?");
        if (state.recentPredictions.length > 20) {
          state.recentPredictions.shift();
        }
        el.tokenTape.textContent = state.recentPredictions.join(" ");
        const currentLoss = state.trainProgress[state.trainProgress.length - 1].loss;
        el.lossLabel.textContent = "Current Loss: " + currentLoss.toFixed(4);
        drawChart();
        await new Promise(function (resolve) {
          setTimeout(resolve, 50);
        });
      } catch (err) {
        console.error(err);
        state.isTraining = false;
      }
    }

    el.startTrainBtn.classList.remove("hidden");
    el.stopTrainBtn.classList.add("hidden");
  }

  function stopTraining() {
    state.isTraining = false;
  }

  async function runInference() {
    const res = await fetch("/api/generate");
    if (!res.ok) {
      throw new Error("inference request failed");
    }
    const data = await res.json();
    el.generatedText.textContent = data.text || "???";
    if (state.traceEnabled) {
      await runInferenceTrace();
    }
  }

  function renderTrace(traceData) {
    el.traceList.innerHTML = "";
    el.traceSummary.textContent = "Stop reason: " + (traceData.stop_reason || "unknown");

    (traceData.steps || []).forEach(function (step) {
      const card = document.createElement("div");
      card.className = "border border-black p-2";

      const title = document.createElement("p");
      title.className = "font-bold";
      title.textContent = "Step " + String(step.position + 1);

      const context = document.createElement("p");
      context.textContent = "Context: " + (step.context || "(empty)");

      const chosen = document.createElement("p");
      chosen.textContent =
        "Chosen token: '" + step.chosen_char + "'  |  p=" + Number(step.chosen_prob || 0).toFixed(4) + "  |  draw=" + Number(step.random_u || 0).toFixed(4);

      const interval = document.createElement("p");
      interval.textContent =
        "Sampling interval: [" + Number(step.cum_before || 0).toFixed(4) + ", " + Number(step.cum_after || 0).toFixed(4) +
        ")  |  Rank by probability: #" + String(step.chosen_rank || "?");

      const reason = document.createElement("p");
      reason.textContent = "Why: " + (step.reason || "");

      const top = document.createElement("p");
      const topText = (step.top_k || []).slice(0, 3).map(function (c) {
        return "'" + c.char + "' " + Number(c.prob || 0).toFixed(4);
      }).join(" | ");
      top.textContent = "Top by probability: " + topText;

      const note = document.createElement("p");
      note.textContent = "Note: top list is sorted by probability; sampling walks full vocabulary index order.";

      card.appendChild(title);
      card.appendChild(context);
      card.appendChild(chosen);
      card.appendChild(interval);
      card.appendChild(top);
      card.appendChild(note);
      card.appendChild(reason);
      el.traceList.appendChild(card);
    });
  }

  async function runInferenceTrace() {
    const res = await fetch("/api/generate_trace", { method: "POST" });
    if (!res.ok) {
      throw new Error("trace inference request failed");
    }
    const data = await res.json();
    el.generatedText.textContent = data.text || "???";
    renderTrace(data);
  }

  function showExplanation(topic) {
    const text = state.explanationByTopic[topic] || "No explanation available.";
    el.explanationText.textContent = text;
    el.explanationBox.classList.remove("hidden");
  }

  function bindEvents() {
    el.tabTheory.addEventListener("click", function () {
      setActiveTab("theory");
    });
    el.tabTrain.addEventListener("click", function () {
      setActiveTab("train");
    });
    el.tabInference.addEventListener("click", function () {
      setActiveTab("inference");
    });

    document.querySelectorAll(".explain-btn").forEach(function (btn) {
      btn.addEventListener("click", function () {
        showExplanation(btn.getAttribute("data-topic"));
      });
    });

    el.startTrainBtn.addEventListener("click", function () {
      startTraining().catch(console.error);
    });
    el.stopTrainBtn.addEventListener("click", stopTraining);
    el.generateBtn.addEventListener("click", function () {
      runInference().catch(console.error);
    });
    el.traceBtn.addEventListener("click", function () {
      if (state.traceEnabled) {
        setTraceEnabled(false);
      } else {
        setTraceEnabled(true);
        runInferenceTrace().catch(console.error);
      }
    });
    el.menuTheory.addEventListener("click", function () {
      setActiveTab("theory");
    });
    el.menuTrain.addEventListener("click", function () {
      setActiveTab("train");
    });
    el.menuInference.addEventListener("click", function () {
      setActiveTab("inference");
    });
    el.addDocBtn.addEventListener("click", function () {
      const value = (el.newDocInput.value || "").trim().toLowerCase();
      if (!value) {
        return;
      }
      state.docs.push(value);
      el.newDocInput.value = "";
      renderDocs();
      initModel().catch(console.error);
    });
    el.newDocInput.addEventListener("keydown", function (evt) {
      if (evt.key === "Enter") {
        el.addDocBtn.click();
      }
    });
  }

  bindEvents();
  renderDocs();
  setActiveTab("theory");
  setTraceEnabled(false);
  initModel().catch(console.error);
})();
