
export interface TrainingStep {
  step: number;
  loss: number;
  learningRate: number;
}

export interface GptConfig {
  n_embd: number;
  n_head: number;
  n_layer: number;
  block_size: number;
  learningRate: number;
}

export interface Tokenizer {
  encode: (text: string) => number[];
  decode: (ids: number[]) => string;
  vocabSize: number;
  chars: string[];
}
