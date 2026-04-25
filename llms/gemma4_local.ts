import { OpenAICompatLlm } from "../llm_api_adapters/openai_compat_adapter.js";

export const gemma4LocalModel = new OpenAICompatLlm({
  model: "google/gemma-4-26b-a4b", // must match exactly what LM Studio shows
  baseUrl: "http://127.0.0.1:1234/v1", // LM Studio OpenAI-compat endpoint
  apiKey: "lm-studio", // ignored by the server, but must be non-empty
  temperature: 0.7,
  maxTokens: 16384,
});
