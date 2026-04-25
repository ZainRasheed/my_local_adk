import { LlmAgent } from '@google/adk';
import { geminiFlashModel } from '../llms/gemini-flash.js';
import { getCurrentTime } from '../tools/agent_function_tools.js';

export const rootAgent = new LlmAgent({
  name: 'hello_time_agent',
  model: geminiFlashModel,
  description: 'Tells the current time in a specified city.',
  instruction: `You are a helpful assistant that tells the current to a city.
                Use the 'get_current_time' tool for this purpose.`,
  tools: [getCurrentTime],
});