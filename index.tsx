
import Alpine from 'alpinejs';
import { GoogleGenAI } from "@google/genai";

// Fix for window.Alpine property error on line 10
(window as any).Alpine = Alpine;

const AI_EXPLAIN_PROMPT = `
You are a Macintosh System 7.0 Guide. Explain Karpathy's Atomic GPT concepts in a friendly, 1990s technical support voice.
The backend is written in Go. Be concise.
`;

Alpine.data('gptApp', () => ({
    activeTab: 'theory',
    docs: ["alex", "james", "mary", "anna", "john", "emily", "luke", "olivia", "noah", "sophia"],
    newDoc: '',
    isTraining: false,
    trainProgress: [] as { step: number; loss: number }[],
    generatedText: '',
    explanation: '',
    isLoadingExplanation: false,
    paramCount: 0,
    isInitialized: false,

    async init() {
        await this.initModel();
        
        // Setup canvas observer
        (this as any).$watch('trainProgress', () => {
            (this as any).drawChart();
        });
    },

    async initModel() {
        const config = {
            n_embd: 16,
            n_head: 4,
            n_layer: 1,
            block_size: 16,
            learningRate: 0.05
        };
        try {
            const res = await fetch('/api/init', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ docs: this.docs, config })
            });
            const data = await res.json();
            this.paramCount = data.params;
            this.isInitialized = true;
            this.trainProgress = [];
        } catch (err) {
            console.error("Init failed", err);
        }
    },

    async addDoc() {
        if (!this.newDoc.trim()) return;
        this.docs.push(this.newDoc.trim().toLowerCase());
        this.newDoc = '';
        await this.initModel();
    },

    removeDoc(index: number) {
        this.docs.splice(index, 1);
        this.initModel();
    },

    async startTraining() {
        if (!this.isInitialized || this.isTraining) return;
        this.isTraining = true;
        
        while (this.isTraining) {
            try {
                const res = await fetch('/api/train', { method: 'POST' });
                if (!res.ok) throw new Error("Server error");
                const data = await res.json();
                
                this.trainProgress.push({ step: data.step, loss: data.loss });
                if (this.trainProgress.length > 50) this.trainProgress.shift();
                
                await new Promise(r => setTimeout(r, 50));
            } catch (err) {
                console.error("Training failed", err);
                this.isTraining = false;
            }
        }
    },

    stopTraining() {
        this.isTraining = false;
    },

    async runInference() {
        try {
            const res = await fetch('/api/generate');
            const data = await res.json();
            this.generatedText = data.text;
        } catch (err) {
            console.error("Inference failed", err);
        }
    },

    async fetchExplanation(topic: string) {
        this.isLoadingExplanation = true;
        try {
            // Using gemini-3-pro-preview for advanced technical/STEM reasoning regarding GPT architecture
            const ai = new GoogleGenAI({ apiKey: process.env.API_KEY as string });
            const response = await ai.models.generateContent({
                model: 'gemini-3-pro-preview',
                contents: `Topic: ${topic}\n\nExplain how this GPT concept works in the context of our Go backend.`,
                config: { systemInstruction: AI_EXPLAIN_PROMPT }
            });
            this.explanation = response.text || 'Error: Could not retrieve info.';
        } catch (err) {
            this.explanation = "System Error: Please check connection.";
        } finally {
            this.isLoadingExplanation = false;
        }
    },

    drawChart() {
        // Fix for errors on lines 123-125: Cast HTMLElement to HTMLCanvasElement to access canvas-specific properties
        const canvas = document.getElementById('lossCanvas') as HTMLCanvasElement | null;
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;
        const w = canvas.width;
        const h = canvas.height;
        
        ctx.clearRect(0, 0, w, h);
        
        if (this.trainProgress.length < 2) return;

        ctx.strokeStyle = 'black';
        ctx.lineWidth = 1;
        ctx.beginPath();
        
        const maxLoss = Math.max(...this.trainProgress.map(p => p.loss), 5);
        
        this.trainProgress.forEach((p, i) => {
            const x = (i / 49) * w;
            const y = h - (p.loss / maxLoss) * h;
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        });
        
        ctx.stroke();
    }
}));

const AppTemplate = `
<div x-data="gptApp" class="p-4 relative h-full">
    <!-- Main Application Window -->
    <div class="mac-window" style="top: 40px; left: 40px; width: 700px; height: 500px;">
        <div class="mac-title-bar">
            <div class="mac-close-box"></div>
            <div class="mac-title-bar-lines"></div>
            <div class="mac-title-text">Atomic GPT Explorer 1.0</div>
        </div>
        
        <div class="mac-tabs px-2 pt-2">
            <div @click="activeTab = 'theory'" :class="{'active': activeTab === 'theory'}" class="mac-tab">Theory</div>
            <div @click="activeTab = 'train'" :class="{'active': activeTab === 'train'}" class="mac-tab">Training</div>
            <div @click="activeTab = 'inference'" :class="{'active': activeTab === 'inference'}" class="mac-tab">Inference</div>
        </div>

        <div class="mac-content">
            <!-- Theory Tab -->
            <template x-if="activeTab === 'theory'">
                <div class="space-y-4">
                    <h2 class="text-lg font-bold">System Architecture</h2>
                    <p class="text-sm italic">Atomic GPT uses a Go-based Autograd engine to compute gradients for a minimalist Transformer.</p>
                    
                    <div class="grid grid-cols-2 gap-4">
                        <button @click="fetchExplanation('Autograd')" class="mac-button">About Autograd</button>
                        <button @click="fetchExplanation('Transformer')" class="mac-button">About Transformer</button>
                        <button @click="fetchExplanation('Attention')" class="mac-button">About Attention</button>
                        <button @click="fetchExplanation('Go Backend')" class="mac-button">About Go Backend</button>
                    </div>

                    <div x-show="explanation" class="border border-black p-3 text-xs bg-slate-50 mt-4">
                        <p class="font-bold mb-1">Guide Explanation:</p>
                        <div x-text="explanation"></div>
                    </div>
                </div>
            </template>

            <!-- Train Tab -->
            <template x-if="activeTab === 'train'">
                <div class="space-y-4">
                    <div class="flex justify-between items-center">
                        <h2 class="text-lg font-bold">Model Training</h2>
                        <div class="flex gap-2">
                            <button x-show="!isTraining" @click="startTraining" class="mac-button">Start Train</button>
                            <button x-show="isTraining" @click="stopTraining" class="mac-button">Stop Train</button>
                        </div>
                    </div>

                    <div class="p-2 border border-black bg-white">
                        <canvas id="lossCanvas" width="600" height="200"></canvas>
                        <div class="flex justify-between text-[10px] mt-1">
                            <span>Step History (Last 50)</span>
                            <span x-text="'Current Loss: ' + (trainProgress.length ? trainProgress[trainProgress.length-1].loss.toFixed(4) : 'N/A')"></span>
                        </div>
                    </div>

                    <div class="grid grid-cols-2 gap-4 text-xs">
                        <div class="border border-black p-2">
                            <p class="font-bold">Model Stats</p>
                            <p>Params: <span x-text="paramCount"></span></p>
                            <p>Language: GoLang 1.21</p>
                        </div>
                        <div class="border border-black p-2">
                            <p class="font-bold">Optimizer</p>
                            <p>Type: Adam</p>
                            <p>Rate: 0.05</p>
                        </div>
                    </div>
                </div>
            </template>

            <!-- Inference Tab -->
            <template x-if="activeTab === 'inference'">
                <div class="space-y-4 flex flex-col items-center">
                    <h2 class="text-lg font-bold w-full">Hallucination Node</h2>
                    
                    <div class="w-full h-32 border-2 border-black bg-white flex items-center justify-center p-4">
                        <div class="code-font text-3xl font-bold" x-text="generatedText || '???'"></div>
                    </div>

                    <button @click="runInference" class="mac-button w-full py-4 text-lg">Generate Next Word</button>
                    
                    <p class="text-[10px] text-center italic mt-4">Note: The model predicts the next character based on patterns learned during the training phase.</p>
                </div>
            </template>
        </div>
    </div>

    <!-- Sidebar / Data Window -->
    <div class="mac-window" style="top: 100px; left: 760px; width: 250px; height: 400px;">
        <div class="mac-title-bar">
            <div class="mac-close-box"></div>
            <div class="mac-title-bar-lines"></div>
            <div class="mac-title-text">Training Data</div>
        </div>
        <div class="mac-content flex flex-col">
            <div class="flex gap-1 mb-2">
                <input x-model="newDoc" @keydown.enter="addDoc" type="text" placeholder="Add entry..." class="mac-input flex-1">
                <button @click="addDoc" class="mac-button">+</button>
            </div>
            <div class="flex-1 border border-black overflow-y-auto">
                <template x-for="(doc, index) in docs" :key="index">
                    <div class="flex justify-between items-center p-1 border-b border-black text-xs hover:bg-black hover:text-white group">
                        <span class="code-font" x-text="doc"></span>
                        <span @click="removeDoc(index)" class="cursor-pointer font-bold px-1">×</span>
                    </div>
                </template>
            </div>
        </div>
    </div>

    <!-- Loading Overlay -->
    <template x-if="isLoadingExplanation">
        <div class="fixed inset-0 bg-white bg-opacity-50 z-[2000] flex items-center justify-center">
            <div class="mac-window p-8 items-center gap-4">
                <div class="text-lg font-bold">Processing...</div>
                <div class="w-16 h-1 bg-black animate-pulse"></div>
            </div>
        </div>
    </template>
</div>
`;

// Added non-null assertion to satisfy TS for getElementById('root')
const rootElement = document.getElementById('root');
if (rootElement) {
    rootElement.innerHTML = AppTemplate;
}
Alpine.start();
