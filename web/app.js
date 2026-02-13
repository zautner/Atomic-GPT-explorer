(function () {
  "use strict";

  const state = {
    activeTab: "theory",
    docs: ["alex", "james", "mary", "anna", "john", "emily", "luke", "olivia", "noah", "sophia", "inna", "vitaly", "daniel", "liza"],
    isTraining: false,
    traceEnabled: false,
    trainProgress: [],
    recentPredictions: [],
    trainOptions: {
      stepsPerCall: 1,
      batchSize: 6
    },
    generateOptions: {
      temperature: 0.7,
      topK: 5,
      minLen: 3
    },
    paramCount: 0,
    isInitialized: false,
    explanationByTopic: {
      "Autograd": "What it is:\nAutograd is an automatic way to compute derivatives. Every math operation creates a node that stores both a value (Data) and how sensitive loss is to that value (Grad).\n\nHow it works here:\n1) Forward pass: the model computes logits and loss.\n2) Backward pass: gradients are propagated from loss back through all connected nodes using the chain rule.\n3) Update: Adam uses those gradients to nudge parameters in directions that should reduce future loss.\n\nWhy it matters:\nWithout autograd, each derivative would need to be hand-coded and easy to break. With autograd, the model can safely combine many operations and still learn from errors.",
      "Transformer": "What it is:\nA Transformer is a next-token predictor built from repeated blocks. Each block mixes information across positions (attention) and then transforms features (MLP).\n\nHow it works in this app:\n1) Character token + position embedding are combined.\n2) RMSNorm stabilizes magnitudes.\n3) Attention reads prior context using query/key/value projections.\n4) Residual connection keeps useful prior signal.\n5) MLP expands then compresses features for richer representation.\n6) Final linear head converts features to logits over possible next characters.\n\nWhy it matters:\nThis structure can model dependencies between distant characters while remaining efficient enough for iterative training and generation in a browser-connected demo.",
      "Attention": "What it is:\nAttention is a weighted lookup over previous tokens. At each step, the model asks: 'Which earlier positions are most relevant to predicting the next character?'\n\nHow the score is formed:\n- Query (current state) is compared with Keys (past states).\n- Dot products are scaled, then softmax converts them into probabilities.\n- Those probabilities weight the Values (past content) to build a context vector.\n\nInterpretation:\nHigher attention weight means that prior token had more influence on this step. In trace output, top candidates reflect the combined effect of these context-aware features plus output projection.",
      "Go Backend": "What it is:\nThe Go backend is the execution engine for model state, math, and APIs. The UI only visualizes data; learning and sampling happen in Go.\n\nRuntime responsibilities:\n- /api/init: build model parameters from docs + config.\n- /api/train: run configurable training work (steps_per_call, batch_size) and return chart/feedback metrics.\n- /api/generate: sample next-token sequence using sampling options.\n- /api/generate_trace: sample once and return step-by-step candidate probabilities and selection reasons.\n\nWhy Go helps here:\nGo gives predictable performance, simple concurrency control via mutexes, and easy deployment as a single binary serving both model APIs and static UI assets."
    },
    htmlByTopic: {
      "Vocabulary": "<div class='space-y-2'>" +
        "<p class='font-bold'>Mini Dictionary (Student Version)</p>" +
        "<p><span class='font-bold'>Token:</span> One character unit the model reads and predicts (for example: 'a', 'n', 'l').</p>" +
        "<p><span class='font-bold'>Vocabulary:</span> The full set of tokens the model is allowed to output.</p>" +
        "<p><span class='font-bold'>Context:</span> Characters already seen so far. The next prediction depends on this context.</p>" +
        "<p><span class='font-bold'>Embedding:</span> A learned list of numbers that represents each token and each position.</p>" +
        "<p><span class='font-bold'>Logit:</span> Raw score for each possible next token before converting to probabilities.</p>" +
        "<p><span class='font-bold'>Probability:</span> Chance of each token after Softmax. All token probabilities add up to 1.00.</p>" +
        "<p><span class='font-bold'>Attention Weight:</span> How strongly the current step listens to an earlier character.</p>" +
        "<p><span class='font-bold'>Loss:</span> Prediction error. Higher loss means worse prediction.</p>" +
        "<p><span class='font-bold'>Gradient:</span> Direction and strength used to change parameters to reduce loss.</p>" +
        "<p><span class='font-bold'>Adam:</span> Optimizer that applies gradients to update parameters each training step.</p>" +
        "<p><span class='font-bold'>Sampling Draw (u):</span> Random number in [0,1) used to pick a token from probabilities.</p>" +
        "<p><span class='font-bold'>Cumulative Interval:</span> Running probability ranges. The interval containing u wins.</p>" +
        "<p><span class='font-bold'>&lt;END&gt; Control Token:</span> In this project one special token is reused for start and stop.</p>" +
        "</div>",
      "Flowchart": "<div class='space-y-3'>" +
        "<p class='font-bold'>Training Flow (per /api/train call)</p>" +
        "<svg viewBox='0 0 780 185' width='100%' height='185' xmlns='http://www.w3.org/2000/svg'>" +
        "<defs><marker id='arr' markerWidth='8' markerHeight='8' refX='7' refY='4' orient='auto'><path d='M0,0 L8,4 L0,8 z' fill='black'/></marker></defs>" +
        "<rect x='10' y='52' width='108' height='40' fill='white' stroke='black'/><text x='64' y='76' text-anchor='middle' font-size='11'>1) Sample</text><text x='64' y='88' text-anchor='middle' font-size='10'>batch_size names</text>" +
        "<line x1='118' y1='72' x2='147' y2='72' stroke='black' marker-end='url(#arr)'/>" +
        "<rect x='147' y='52' width='116' height='40' fill='white' stroke='black'/><text x='205' y='76' text-anchor='middle' font-size='11'>2) Encode</text><text x='205' y='88' text-anchor='middle' font-size='10'>chars to IDs</text>" +
        "<line x1='263' y1='72' x2='292' y2='72' stroke='black' marker-end='url(#arr)'/>" +
        "<rect x='292' y='52' width='122' height='40' fill='white' stroke='black'/><text x='353' y='76' text-anchor='middle' font-size='11'>3) Forward</text><text x='353' y='88' text-anchor='middle' font-size='10'>predict next char</text>" +
        "<line x1='414' y1='72' x2='443' y2='72' stroke='black' marker-end='url(#arr)'/>" +
        "<rect x='443' y='52' width='96' height='40' fill='white' stroke='black'/><text x='491' y='76' text-anchor='middle' font-size='11'>4) Loss</text><text x='491' y='88' text-anchor='middle' font-size='10'>prediction error</text>" +
        "<line x1='539' y1='72' x2='568' y2='72' stroke='black' marker-end='url(#arr)'/>" +
        "<rect x='568' y='52' width='96' height='40' fill='white' stroke='black'/><text x='616' y='76' text-anchor='middle' font-size='11'>5) Backward</text><text x='616' y='88' text-anchor='middle' font-size='10'>compute grads</text>" +
        "<line x1='664' y1='72' x2='693' y2='72' stroke='black' marker-end='url(#arr)'/>" +
        "<rect x='693' y='52' width='76' height='40' fill='white' stroke='black'/><text x='731' y='76' text-anchor='middle' font-size='11'>6) Adam</text><text x='731' y='88' text-anchor='middle' font-size='10'>update</text>" +
        "<path d='M731 94 L731 130 L64 130 L64 95' fill='none' stroke='black' marker-end='url(#arr)'/>" +
        "<text x='398' y='146' text-anchor='middle' font-size='10'>Loop repeats for steps_per_call. More good data + enough steps usually lowers loss.</text>" +
        "</svg>" +
        "<p class='font-bold'>Inference Sampling (example probabilities sum to 1.00)</p>" +
        "<svg viewBox='0 0 780 165' width='100%' height='165' xmlns='http://www.w3.org/2000/svg'>" +
        "<rect x='0' y='0' width='780' height='165' fill='white' stroke='black'/>" +
        "<line x1='45' y1='120' x2='520' y2='120' stroke='black'/>" +
        "<line x1='45' y1='20' x2='45' y2='120' stroke='black'/>" +
        "<rect x='80' y='62' width='52' height='58' fill='#0a7a0a' stroke='black'/><text x='106' y='136' text-anchor='middle' font-size='10'>a</text><text x='106' y='56' text-anchor='middle' font-size='10'>0.58</text>" +
        "<rect x='155' y='97' width='52' height='23' fill='#b58900' stroke='black'/><text x='181' y='136' text-anchor='middle' font-size='10'>n</text><text x='181' y='91' text-anchor='middle' font-size='10'>0.23</text>" +
        "<rect x='230' y='109' width='52' height='11' fill='#b58900' stroke='black'/><text x='256' y='136' text-anchor='middle' font-size='10'>i</text><text x='256' y='103' text-anchor='middle' font-size='10'>0.11</text>" +
        "<rect x='305' y='115' width='52' height='5' fill='#b00020' stroke='black'/><text x='331' y='136' text-anchor='middle' font-size='10'>e</text><text x='331' y='109' text-anchor='middle' font-size='10'>0.05</text>" +
        "<rect x='380' y='117' width='52' height='3' fill='#b00020' stroke='black'/><text x='406' y='136' text-anchor='middle' font-size='10'>&lt;END&gt;</text><text x='406' y='111' text-anchor='middle' font-size='10'>0.03</text>" +
        "<text x='555' y='40' font-size='11'>Draw u = 0.64</text>" +
        "<text x='555' y='58' font-size='10'>Cumulative intervals:</text>" +
        "<text x='555' y='74' font-size='10'>a: [0.00, 0.58)</text>" +
        "<text x='555' y='90' font-size='10'>n: [0.58, 0.81)  &lt;- chosen</text>" +
        "<text x='555' y='106' font-size='10'>i: [0.81, 0.92), e: [0.92, 0.97)</text>" +
        "<text x='555' y='122' font-size='10'>&lt;END&gt;: [0.97, 1.00)</text>" +
        "</svg>" +
        "</div>"
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
    addDocBtn: document.getElementById("addDocBtn"),
    trainStepsInput: document.getElementById("trainStepsInput"),
    trainBatchInput: document.getElementById("trainBatchInput"),
    tempInput: document.getElementById("tempInput"),
    topKInput: document.getElementById("topKInput"),
    minLenInput: document.getElementById("minLenInput"),
    presetSafeBtn: document.getElementById("presetSafeBtn"),
    presetCreativeBtn: document.getElementById("presetCreativeBtn"),
    presetRandomBtn: document.getElementById("presetRandomBtn")
  };

  function clampNumber(value, min, max, fallback) {
    const n = Number(value);
    if (!Number.isFinite(n)) {
      return fallback;
    }
    return Math.min(max, Math.max(min, n));
  }

  function readTrainOptions() {
    state.trainOptions.stepsPerCall = Math.round(
      clampNumber(el.trainStepsInput.value, 1, 20, 1)
    );
    state.trainOptions.batchSize = Math.round(
      clampNumber(el.trainBatchInput.value, 1, 64, 6)
    );
    el.trainStepsInput.value = String(state.trainOptions.stepsPerCall);
    el.trainBatchInput.value = String(state.trainOptions.batchSize);
  }

  function readGenerateOptions() {
    state.generateOptions.temperature = clampNumber(el.tempInput.value, 0.1, 2, 0.7);
    state.generateOptions.topK = Math.round(
      clampNumber(el.topKInput.value, 1, 64, 5)
    );
    state.generateOptions.minLen = Math.round(
      clampNumber(el.minLenInput.value, 0, 32, 3)
    );
    el.tempInput.value = String(state.generateOptions.temperature);
    el.topKInput.value = String(state.generateOptions.topK);
    el.minLenInput.value = String(state.generateOptions.minLen);
  }

  function setSamplingPreset(kind) {
    if (kind === "safe") {
      el.tempInput.value = "0.6";
      el.topKInput.value = "4";
      el.minLenInput.value = "3";
    } else if (kind === "creative") {
      el.tempInput.value = "0.9";
      el.topKInput.value = "8";
      el.minLenInput.value = "2";
    } else if (kind === "random") {
      el.tempInput.value = "1.6";
      el.topKInput.value = "16";
      el.minLenInput.value = "1";
    }
    readGenerateOptions();
  }

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
        readTrainOptions();
        const res = await fetch("/api/train", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            steps_per_call: state.trainOptions.stepsPerCall,
            batch_size: state.trainOptions.batchSize
          })
        });
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
    if (state.traceEnabled) {
      await runInferenceTrace();
      return;
    }

    readGenerateOptions();
    const res = await fetch("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ options: {
        temperature: state.generateOptions.temperature,
        top_k: state.generateOptions.topK,
        min_len: state.generateOptions.minLen
      } })
    });
    if (!res.ok) {
      throw new Error("inference request failed");
    }
    const data = await res.json();
    el.generatedText.textContent = data.text || "???";
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
    readGenerateOptions();
    const res = await fetch("/api/generate_trace", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ options: {
        temperature: state.generateOptions.temperature,
        top_k: state.generateOptions.topK,
        min_len: state.generateOptions.minLen
      } })
    });
    if (!res.ok) {
      throw new Error("trace inference request failed");
    }
    const data = await res.json();
    el.generatedText.textContent = data.text || "???";
    renderTrace(data);
  }

  function showExplanation(topic) {
    if (state.htmlByTopic[topic]) {
      el.explanationText.innerHTML = state.htmlByTopic[topic];
      return;
    }
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
    el.trainStepsInput.addEventListener("change", readTrainOptions);
    el.trainBatchInput.addEventListener("change", readTrainOptions);
    el.tempInput.addEventListener("change", readGenerateOptions);
    el.topKInput.addEventListener("change", readGenerateOptions);
    el.minLenInput.addEventListener("change", readGenerateOptions);
    el.presetSafeBtn.addEventListener("click", function () {
      setSamplingPreset("safe");
    });
    el.presetCreativeBtn.addEventListener("click", function () {
      setSamplingPreset("creative");
    });
    el.presetRandomBtn.addEventListener("click", function () {
      setSamplingPreset("random");
    });
  }

  bindEvents();
  renderDocs();
  readTrainOptions();
  readGenerateOptions();
  setActiveTab("theory");
  setTraceEnabled(false);
  showExplanation("Vocabulary");
  initModel().catch(console.error);
})();
