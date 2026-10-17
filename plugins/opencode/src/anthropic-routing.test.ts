import { once } from "node:events";
import http from "node:http";
import { networkInterfaces } from "node:os";
import { promisify } from "node:util";
import { AI } from "@opencode/ai/promise";
import { model as anthropicModel } from "@opencode/ai/providers/anthropic";
import type { Plugin } from "@opencode/plugin";
import { expect, it } from "vitest";
import { z } from "zod";

import { headroomSetup } from "./plugin.js";

interface FixtureServer {
  url: string;
  close(): Promise<void>;
}

async function fixtureServer(host: string, handler: http.RequestListener): Promise<FixtureServer> {
  const server = http.createServer(handler);
  const listening = once(server, "listening");
  server.listen(0, host);
  await listening;
  const address = server.address();
  if (!address || typeof address === "string") throw new Error("Expected TCP server address");
  return {
    url: `http://${host}:${address.port}`,
    async close() {
      server.closeAllConnections();
      await promisify(server.close.bind(server))();
    },
  };
}

const requestSchema = z.object({
  model: z.string(),
  messages: z.array(z.object({
    content: z.array(z.object({ type: z.literal("text"), text: z.string() })),
  })),
});

async function readBody(request: http.IncomingMessage): Promise<string> {
  request.setEncoding("utf8");
  let body = "";
  for await (const chunk of request) {
    if (typeof chunk !== "string") throw new TypeError("Expected UTF-8 request body");
    body += chunk;
  }
  return body;
}

function replyFrames(model: string, text: string): string {
  const frames = [
    ["message_start", { type: "message_start", message: { id: "msg_fixture", type: "message", role: "assistant", model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: 10, output_tokens: 0 } } }],
    ["content_block_start", { type: "content_block_start", index: 0, content_block: { type: "text", text: "" } }],
    ["content_block_delta", { type: "content_block_delta", index: 0, delta: { type: "text_delta", text } }],
    ["content_block_stop", { type: "content_block_stop", index: 0 }],
    ["message_delta", { type: "message_delta", delta: { stop_reason: "end_turn", stop_sequence: null }, usage: { output_tokens: 1 } }],
    ["message_stop", { type: "message_stop" }],
  ] as const;
  return frames.map(([event, data]) => `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`).join("");
}

interface CatalogModel {
  id: string;
  providerID: string;
  package?: string;
  settings: { baseURL?: string; authToken: string };
  headers: Record<string, string>;
}

interface CatalogEditor {
  list(): CatalogModel[];
  provider: { get(): { provider: { package: string; settings: { baseURL: string } } } };
  update(provider: string, id: string, mutate: (entry: CatalogModel) => void): void;
}

it.for(["provider-package", "model-package", "local-model"] as const)(
  "preserves native Messages inference after Effect captures fetch: %s",
  async (kind, test) => {
    const host = kind === "local-model" ? "127.0.0.1" : Object.values(networkInterfaces())
      .flat().find((entry) => entry && !entry.internal && entry.family === "IPv4")?.address;
    if (!host) {
      test.skip();
      return;
    }
    const capturedFetch = globalThis.fetch.bind(globalThis);
    let upstreamRequests = 0;
    let proxyRequests = 0;
    const upstream = await fixtureServer(host, async (request, response) => {
      if (new URL(request.url ?? "/", "http://fixture").pathname !== "/v1/messages") {
        response.writeHead(404).end();
        return;
      }
      if (request.headers.authorization !== "Bearer synthetic-native-oauth") {
        response.writeHead(401).end();
        return;
      }
      upstreamRequests++;
      const body = requestSchema.parse(JSON.parse(await readBody(request)));
      const values = z.array(z.number()).parse(JSON.parse(body.messages[0].content[0].text));
      response.writeHead(200, { "content-type": "text/event-stream" });
      response.end(replyFrames(body.model, String(values.reduce((a, b) => a + b, 0))));
    });
    const proxy = await fixtureServer("127.0.0.1", async (request, response) => {
      proxyRequests++;
      const target = request.headers["x-headroom-base-url"];
      const path = request.headers["x-headroom-original-path"];
      if (target !== upstream.url || path !== "/v1/messages") {
        response.writeHead(400).end();
        return;
      }
      const headers = new Headers();
      for (const [name, value] of Object.entries(request.headers)) {
        if (typeof value === "string" && name !== "host") headers.set(name, value);
      }
      const result = await capturedFetch(target + path + new URL(request.url ?? "/", proxy.url).search, {
        method: "POST", headers, body: await readBody(request),
      });
      response.writeHead(result.status, { "content-type": "text/event-stream" });
      response.end(await result.text());
    });
    const entry: CatalogModel = {
      id: "claude-sonnet-4-5", providerID: "github-copilot",
      package: kind === "model-package" ? "@opencode/ai/providers/anthropic" : undefined,
      settings: { authToken: "synthetic-native-oauth" }, headers: {},
    };
    const provider = {
      package: kind === "model-package" ? "@opencode/ai/providers/openai-compatible" : "@opencode/ai/providers/anthropic",
      settings: { baseURL: upstream.url + "/v1" },
    };
    const hostContext = {
      options: { proxyUrl: proxy.url },
      location: { directory: "/fixture", project: { id: "native-routing" } },
      model: { async transform(callback: (editor: CatalogEditor) => void) {
        callback({ list: () => [entry], provider: { get: () => ({ provider }) },
          update(_provider, _id, mutate) { mutate(entry); } });
      } },
      tool: { async transform(callback: (editor: { add(tool: unknown): void }) => void) { callback({ add() {} }); } },
      shell: { async hook(_name: string, _callback: unknown) {} },
    };
    // Only host registration is controlled; inference uses the real SDK and sockets.
    const context = hostContext as unknown as Plugin.Context;
    const ai = AI.make();
    let cleanup: void | (() => void | Promise<void>) = undefined;
    try {
      await ai.llm.generate({ model: anthropicModel(entry.id, { ...provider.settings, ...entry.settings }),
        prompt: "[1,2]", generation: { maxTokens: 32 } });
      upstreamRequests = 0;
      cleanup = await headroomSetup(context);
      const result = await ai.llm.generate({ model: anthropicModel(entry.id, { ...provider.settings, ...entry.settings, headers: entry.headers }),
        prompt: "[12,34,98]", generation: { maxTokens: 32 } });
      expect(result.text).toBe("144");
      expect(proxyRequests).toBe(kind === "local-model" ? 0 : 1);
      expect(upstreamRequests).toBe(1);
    } finally {
      await cleanup?.();
      await ai.dispose();
      await proxy.close();
      await upstream.close();
    }
  },
);
