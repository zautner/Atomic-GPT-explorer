(function () {
  "use strict";

  const state = {
    activeTab: "theory",
    docs: ["alex", "james", "mary", "anna", "john", "emily", "luke", "olivia", "noah", "sophia"],
    isTraining: false,
    trainProgress: [],
    paramCount: 0,
    isInitialized: false,
    explanationByTopic: {
      "Autograd": "Autograd tracks every math operation as a graph of Values. During backpropagation, each Value receives gradients that show how much it affected loss.",
      "Transformer": "The model combines token and position embeddings, applies attention and MLP layers, then projects to logits for next-character prediction.",
      "Attention": "Attention computes query-key similarity, turns scores into weights with softmax, and builds context vectors as weighted sums of value vectors.",
      "Go Backend": "The Go backend runs training and inference endpoints, updates weights with Adam, and serializes JSON responses for the browser."
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
    tracePanel: document.getElementById("tracePanel"),
    traceSummary: document.getElementById("traceSummary"),
    traceList: document.getElementById("traceList"),
    lossCanvas: document.getElementById("lossCanvas"),
    lossLabel: document.getElementById("lossLabel"),
    paramCount: document.getElementById("paramCount"),
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
    el.paramCount.textContent = String(state.paramCount);
    el.lossLabel.textContent = "Current Loss: N/A";
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

    ctx.strokeStyle = "black";
    ctx.lineWidth = 1;
    ctx.beginPath();

    const maxLoss = Math.max.apply(
      null,
      state.trainProgress.map(function (p) {
        return p.loss;
      }).concat([5])
    );

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
        state.trainProgress.push({ step: data.step, loss: data.loss });
        if (state.trainProgress.length > 50) {
          state.trainProgress.shift();
        }
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
    el.traceList.innerHTML = "";
    el.traceSummary.textContent = "Trace cleared. Press \"Explain Choice\" to inspect token selection.";
  }

  function renderTrace(traceData) {
    el.traceList.innerHTML = "";
    el.traceSummary.textContent = "Stop reason: " + (traceData.stop_reason || "unknown");

    (traceData.steps || []).forEach(function (step) {
      const card = document.createElement("div");
      card.className = "border border-black p-2";

      const title = document.createElement("p");
      title.className = "font-bold";
      title.textContent = "Step " + String(step.position + 1) + " | Context: " + (step.context || "(empty)");

      const chosen = document.createElement("p");
      chosen.textContent =
        "Chosen: '" + step.chosen_char + "' (p=" + Number(step.chosen_prob || 0).toFixed(4) + "), random_u=" + Number(step.random_u || 0).toFixed(4);

      const reason = document.createElement("p");
      reason.textContent = step.reason || "";

      const top = document.createElement("p");
      const topText = (step.top_k || []).map(function (c) {
        return "'" + c.char + "' " + Number(c.prob || 0).toFixed(4);
      }).join(" | ");
      top.textContent = "Top options: " + topText;

      card.appendChild(title);
      card.appendChild(chosen);
      card.appendChild(top);
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
      runInferenceTrace().catch(console.error);
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
  initModel().catch(console.error);
})();
