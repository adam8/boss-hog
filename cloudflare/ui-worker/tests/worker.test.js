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
    expect(await response.text()).toContain("Private RBP Hog Explorer");
  });

  it("serves the browser assets from the worker bundle", async () => {
    const response = await worker.fetch(new Request("https://example.com/app/app.js"), env);
    expect(response.status).toBe(200);
    expect(response.headers.get("content-type")).toContain("text/javascript");
    expect(await response.text()).toContain("runBacktest");
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
