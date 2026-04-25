/**
 * OpenAI-compatible endpoint via native fetch (Node ≥18 / v24 built-in).
 * OpenAICompatLlm — ADK BaseLlm adapter for any OpenAI-compatible endpoint.
 *
 * Translates between ADK's internal Content/LlmRequest/LlmResponse format
 * and the OpenAI Chat Completions API wire format, using native fetch.
 * Works with LM Studio, Ollama, vLLM, llama.cpp, or any OpenAI-compat server.
 *
 * Usage:
 *   const model = new OpenAICompatLlm({
 *     model: 'google/gemma-4-26b-a4b',
 *     baseUrl: 'http://127.0.0.1:1234/v1',  // LM Studio default
 *     apiKey: 'lm-studio',   // ignored by most local servers, but required non-empty
 *   });
 */

import { BaseLlm } from '@google/adk';
import type {
  BaseLlmConnection,
  LlmRequest,
  LlmResponse,
} from '@google/adk';
import type { Content, Part } from '@google/genai';

// ── OpenAI wire-types (only what we need) ─────────────────────────────────────

interface OAIMessage {
  role: 'system' | 'user' | 'assistant' | 'tool';
  content: string | null;
  tool_calls?: OAIToolCall[];
  tool_call_id?: string;
  name?: string;
}

interface OAIToolCall {
  id: string;
  type: 'function';
  function: { name: string; arguments: string };
}

interface OAIChoice {
  message: OAIMessage;
  finish_reason: string;
}

interface OAIResponse {
  choices: OAIChoice[];
  usage?: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
}

interface OAIStreamChunk {
  choices: Array<{
    delta: Partial<OAIMessage>;
    finish_reason: string | null;
  }>;
}

// ── Converter: ADK Content[] → OpenAI messages[] ──────────────────────────────

function partsToText(parts: Part[]): string {
  return parts
    .map((p) => {
      if ('text' in p && typeof p.text === 'string') return p.text;
      return '';
    })
    .join('');
}

function adkContentsToOAIMessages(
  contents: Content[],
  systemInstruction?: string,
): OAIMessage[] {
  const messages: OAIMessage[] = [];

  if (systemInstruction) {
    messages.push({ role: 'system', content: systemInstruction });
  }

  for (const content of contents) {
    const role = content.role === 'model' ? 'assistant' : 'user';
    const parts = content.parts ?? [];

    // Collect function calls (assistant tool_calls)
    const toolCalls: OAIToolCall[] = [];
    const textParts: string[] = [];
    const funcResponses: Array<{ name: string; id: string; output: string }> =
      [];

    for (const part of parts) {
      if ('functionCall' in part && part.functionCall) {
        toolCalls.push({
          id: part.functionCall.id ?? `call_${part.functionCall.name}`,
          type: 'function',
          function: {
            name: part.functionCall.name!,
            arguments: JSON.stringify(part.functionCall.args ?? {}),
          },
        });
      } else if ('functionResponse' in part && part.functionResponse) {
        funcResponses.push({
          name: part.functionResponse.name!,
          id: part.functionResponse.id ?? `call_${part.functionResponse.name}`,
          output: JSON.stringify(part.functionResponse.response),
        });
      } else if ('text' in part && typeof part.text === 'string') {
        textParts.push(part.text);
      }
    }

    if (funcResponses.length > 0) {
      // Tool results come as 'tool' role messages
      for (const r of funcResponses) {
        messages.push({
          role: 'tool',
          tool_call_id: r.id,
          name: r.name,
          content: r.output,
        });
      }
    } else if (toolCalls.length > 0) {
      messages.push({
        role: 'assistant',
        content: textParts.join('') || null,
        tool_calls: toolCalls,
      });
    } else {
      messages.push({ role, content: textParts.join('') });
    }
  }

  return messages;
}

// ── Converter: OpenAI response → ADK LlmResponse ─────────────────────────────

function oaiChoiceToLlmResponse(choice: OAIChoice): LlmResponse {
  const msg = choice.message;
  const parts: Part[] = [];

  if (msg.content) {
    parts.push({ text: msg.content });
  }

  if (msg.tool_calls) {
    for (const tc of msg.tool_calls) {
      let args: Record<string, unknown> = {};
      try {
        args = JSON.parse(tc.function.arguments);
      } catch {
        args = { raw: tc.function.arguments };
      }
      parts.push({
        functionCall: {
          id: tc.id,
          name: tc.function.name,
          args,
        },
      });
    }
  }

  const content: Content = { role: 'model', parts };
  return { content };
}

// ── Build OpenAI tools from ADK request ──────────────────────────────────────

function buildOAITools(llmRequest: LlmRequest): object[] | undefined {
  const toolsDict = llmRequest.toolsDict ?? {};
  const allowed = llmRequest.allowedTools;

  const entries = Object.entries(toolsDict).filter(
    ([name]) => !allowed || allowed.includes(name),
  );

  if (entries.length === 0) return undefined;

  return entries.map(([_name, tool]) => {
    // BaseTool exposes its JSON schema via tool.declaration
    const decl = (tool as any).declaration?.();
    return {
      type: 'function',
      function: {
        name: decl?.name ?? _name,
        description: decl?.description ?? '',
        parameters: decl?.parameters ?? { type: 'object', properties: {} },
      },
    };
  });
}

// ── OpenAICompatLlm ──────────────────────────────────────────────────────────

export interface OpenAICompatLlmParams {
  /** Model ID exactly as the server expects (e.g. 'google/gemma-4-26b-a4b') */
  model: string;
  /** Base URL of the OpenAI-compat server, e.g. 'http://127.0.0.1:1234/v1' */
  baseUrl?: string;
  /** API key — most local servers ignore it, but it must be non-empty */
  apiKey?: string;
  /** Request temperature (default 0.7) */
  temperature?: number;
  /** Max tokens to generate (default 2048) */
  maxTokens?: number;
}

/** @deprecated Use OpenAICompatLlm */
export { OpenAICompatLlm as LmStudioLlm };

export class OpenAICompatLlm extends BaseLlm {
  static readonly supportedModels: Array<string | RegExp> = [
    // Matches any model string — you can narrow this if needed
    /.*/,
  ];

  private readonly baseUrl: string;
  private readonly apiKey: string;
  private readonly temperature: number;
  private readonly maxTokens: number;

  constructor({
    model,
    baseUrl = 'http://127.0.0.1:1234/v1',
    apiKey = 'lm-studio',
    temperature = 0.7,
    maxTokens = 2048,
  }: OpenAICompatLlmParams) {
    super({ model });
    this.baseUrl = baseUrl.replace(/\/$/, ''); // strip trailing slash
    this.apiKey = apiKey;
    this.temperature = temperature;
    this.maxTokens = maxTokens;
  }

  // ── generateContentAsync ────────────────────────────────────────────────────

  async *generateContentAsync(
    llmRequest: LlmRequest,
    stream = false,
    _abortSignal?: AbortSignal,
  ): AsyncGenerator<LlmResponse, void> {
    this.maybeAppendUserContent(llmRequest);

    // Extract system instruction from config
    const systemInstruction =
      typeof llmRequest.config?.systemInstruction === 'string'
        ? llmRequest.config.systemInstruction
        : llmRequest.config?.systemInstruction
          ? partsToText(
              (llmRequest.config.systemInstruction as Content).parts ?? [],
            )
          : undefined;

    const messages = adkContentsToOAIMessages(
      llmRequest.contents,
      systemInstruction,
    );

    const tools = buildOAITools(llmRequest);

    const body: Record<string, unknown> = {
      model: llmRequest.model ?? this.model,
      messages,
      temperature: this.temperature,
      max_tokens: this.maxTokens,
      stream,
    };

    if (tools && tools.length > 0) {
      body.tools = tools;
      body.tool_choice = 'auto';
    }

    const url = `${this.baseUrl}/chat/completions`;

    if (stream) {
      yield* this._streamCompletion(url, body, _abortSignal);
    } else {
      yield await this._completeOnce(url, body, _abortSignal);
    }
  }

  // ── Single (non-streaming) request ─────────────────────────────────────────

  private async _completeOnce(
    url: string,
    body: Record<string, unknown>,
    signal?: AbortSignal,
  ): Promise<LlmResponse> {
    const res = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${this.apiKey}`,
      },
      body: JSON.stringify(body),
      signal,
    });

    if (!res.ok) {
      const text = await res.text();
      return {
        errorCode: String(res.status),
        errorMessage: `OpenAI-compat server error ${res.status}: ${text}`,
      };
    }

    const json = (await res.json()) as OAIResponse;
    if (!json.choices?.length) {
      return { errorCode: 'NO_CHOICES', errorMessage: 'No choices returned' };
    }

    return oaiChoiceToLlmResponse(json.choices[0]);
  }

  // ── Streaming request ───────────────────────────────────────────────────────

  private async *_streamCompletion(
    url: string,
    body: Record<string, unknown>,
    signal?: AbortSignal,
  ): AsyncGenerator<LlmResponse, void> {
    const res = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${this.apiKey}`,
      },
      body: JSON.stringify(body),
      signal,
    });

    if (!res.ok || !res.body) {
      const text = await res.text();
      yield {
        errorCode: String(res.status),
        errorMessage: `OpenAI-compat server error ${res.status}: ${text}`,
      };
      return;
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = '';
    let accText = '';
    const accToolCalls: Map<number, OAIToolCall> = new Map();

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buf += decoder.decode(value, { stream: true });
      const lines = buf.split('\n');
      buf = lines.pop() ?? '';

      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed || trimmed === 'data: [DONE]') continue;
        if (!trimmed.startsWith('data: ')) continue;

        let chunk: OAIStreamChunk;
        try {
          chunk = JSON.parse(trimmed.slice(6));
        } catch {
          continue;
        }

        const delta = chunk.choices?.[0]?.delta;
        if (!delta) continue;

        const finishReason = chunk.choices?.[0]?.finish_reason;

        if (delta.content) {
          accText += delta.content;
          yield {
            content: { role: 'model', parts: [{ text: delta.content }] },
            partial: true,
          };
        }

        // Accumulate streamed tool calls
        if (delta.tool_calls) {
          for (const tc of delta.tool_calls) {
            const idx = (tc as any).index ?? 0;
            if (!accToolCalls.has(idx)) {
              accToolCalls.set(idx, {
                id: tc.id ?? '',
                type: 'function',
                function: { name: tc.function?.name ?? '', arguments: '' },
              });
            }
            const existing = accToolCalls.get(idx)!;
            if (tc.id) existing.id = tc.id;
            if (tc.function?.name) existing.function.name = tc.function.name;
            if (tc.function?.arguments)
              existing.function.arguments += tc.function.arguments;
          }
        }

        if (finishReason) {
          // Emit final message with all accumulated tool calls
          const finalParts: Part[] = [];
          if (accText) finalParts.push({ text: accText });

          for (const tc of accToolCalls.values()) {
            let args: Record<string, unknown> = {};
            try {
              args = JSON.parse(tc.function.arguments);
            } catch {
              args = { raw: tc.function.arguments };
            }
            finalParts.push({
              functionCall: { id: tc.id, name: tc.function.name, args },
            });
          }

          if (finalParts.length > 0) {
            yield {
              content: { role: 'model', parts: finalParts },
              turnComplete: true,
            };
          }
        }
      }
    }
  }

  // ── connect — live/streaming sessions not supported ────────────────────────

  async connect(_llmRequest: LlmRequest): Promise<BaseLlmConnection> {
    throw new Error(
      'OpenAICompatLlm does not support live (WebSocket) connections. ' +
        'Use generateContentAsync instead.',
    );
  }
}
