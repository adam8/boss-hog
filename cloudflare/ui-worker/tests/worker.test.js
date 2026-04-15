import { describe, expect, it, vi } from "vitest";

import worker from "../src/index.js";

describe("ui worker", () => {
  const env = {
    REQUIRE_ACCESS_VALIDATION: "false",
    HOG_API: {
      fetch: vi.fn(async () =>
        new Response(JSON.stringify({ ok: true }), {
          status: 200,
          headers: { "content-type": "application/json; charset=utf-8" },
        }),
      ),
    },
  };

  it("serves the index document", async () => {
    const response = await worker.fetch(new Request("https://example.com/"), env);
    expect(response.status).toBe(200);
    expect(response.headers.get("content-type")).toContain("text/html");
    const body = await response.text();
    expect(body).toContain("Hog Price Prediction Explorer");
    expect(body).toContain("/app/logo.png");
  });

  it("serves the browser assets from the worker bundle", async () => {
    const response = await worker.fetch(new Request("https://example.com/app/app.js"), env);
    expect(response.status).toBe(200);
    expect(response.headers.get("content-type")).toContain("text/javascript");
    expect(await response.text()).toContain("runBacktest");
  });

  it("serves the logo asset from the worker bundle", async () => {
    const response = await worker.fetch(new Request("https://example.com/app/logo.png"), env);
    expect(response.status).toBe(200);
    expect(response.headers.get("content-type")).toContain("image/png");
    expect((await response.arrayBuffer()).byteLength).toBeGreaterThan(0);
  });

  it("proxies API requests through the service binding", async () => {
    const response = await worker.fetch(
      new Request("https://example.com/api/backtest?feature_pack=price_only"),
      env,
    );
    expect(env.HOG_API.fetch).toHaveBeenCalledOnce();
    expect(response.status).toBe(200);
    expect(await response.text()).toContain("\"ok\":true");
  });
});
