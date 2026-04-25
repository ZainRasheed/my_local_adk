import { LlmAgent } from "@google/adk";
import { gemma4LocalModel } from "../llms/gemma4_local.js";
import { getCurrentTime } from '../tools/agent_function_tools.js';

/* ── 1. Register your local LM Studio model ───────────────────────────────────────── */
const localModel = gemma4LocalModel;

/* ── 3. Build the agent ──────────────────────────────────────────────────── */
export const rootAgent = new LlmAgent({
  name: "hello_time_agent",
  model: localModel, // pass the OpenAICompatLlm instance, not a string
  description: "Tells the current time in a specified city.",
  instruction: `You are a helpful assistant that tells the current time in a city.
                Use the 'get_current_time' tool for this purpose.`,
  tools: [getCurrentTime],
});
