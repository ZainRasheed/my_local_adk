/**
 * run_agent_inmemory_chatbot.ts — Interactive terminal chatbot.
 * Uses OpenAICompatLlm (ADK bridge) → LM Studio → gemma-4-26b-a4b.
 *
 * Run with:
 *  npm run run
 *
 * Run with:
 *   npx tsx run_agent_inmemory_chatbot.ts
 *
 */

import * as readline from "readline";
import { InMemoryRunner } from "@google/adk";
import { rootAgent } from "./agents/test_local_gemma4_agent.js";

const RESET = "\x1b[0m";
const CYAN = "\x1b[36m";
const GREEN = "\x1b[32m";
const YELLOW = "\x1b[33m";
const DIM = "\x1b[2m";

async function main() {
  const runner = new InMemoryRunner({
    agent: rootAgent,
    appName: "lm-studio-demo",
  });

  const session = await runner.sessionService.createSession({
    appName: "lm-studio-demo",
    userId: "user-1",
  });

  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
    terminal: true,
  });

  console.log(`\n${CYAN}╔════════════════════════════════════════╗`);
  console.log(`║  LM Studio Chatbot  (gemma-4-26b-a4b)  ║`);
  console.log(`╚════════════════════════════════════════╝${RESET}`);
  console.log(
    `${DIM}Type your message and press Enter. Type 'exit' or Ctrl+C to quit.\n${RESET}`,
  );

  const ask = () => {
    rl.question(`${GREEN}You: ${RESET}`, async (userMessage) => {
      const trimmed = userMessage.trim();

      if (!trimmed) {
        ask();
        return;
      }
      if (
        trimmed.toLowerCase() === "exit" ||
        trimmed.toLowerCase() === "quit"
      ) {
        console.log(`\n${DIM}Goodbye!${RESET}\n`);
        rl.close();
        return;
      }

      // Pause prompt while the agent is thinking
      rl.pause();
      process.stdout.write(`${YELLOW}Agent: ${RESET}`);

      try {
        const events = runner.runAsync({
          userId: session.userId,
          sessionId: session.id,
          newMessage: {
            role: "user",
            parts: [{ text: trimmed }],
          },
        });

        let wroteAnything = false;

        for await (const event of events) {
          // Skip non-model events (tool calls, internal events, etc.)
          if (event.content?.role !== "model") continue;

          for (const part of event.content.parts ?? []) {
            if ("text" in part && part.text) {
              process.stdout.write(part.text);
              wroteAnything = true;
            }
          }
        }

        if (!wroteAnything) {
          process.stdout.write(`${DIM}(no response)${RESET}`);
        }

        process.stdout.write("\n\n");
      } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : String(err);
        process.stdout.write(`\n${DIM}[Error: ${msg}]${RESET}\n\n`);
      }

      rl.resume();
      ask(); // next turn
    });
  };

  // Handle Ctrl+C gracefully
  rl.on("close", () => {
    process.exit(0);
  });

  ask();
}

main().catch((err) => {
  console.error("Fatal error:", err);
  process.exit(1);
});
