import { beforeEach, describe, expect, it, vi } from "vitest";

describe("requireAccess", () => {
  beforeEach(async () => {
    vi.resetModules();
    vi.stubGlobal("fetch", vi.fn(async () => ({
      ok: true,
      json: async () => ({
        keys: [{ kid: "kid-1", kty: "RSA", e: "AQAB", n: "abc", alg: "RS256" }],
      }),
    })));
    vi.stubGlobal("crypto", {
      subtle: {
        importKey: vi.fn(async () => "key"),
        verify: vi.fn(async () => true),
      },
    });
  });

  it("returns null when access validation passes", async () => {
    const { requireAccess, resetVerifierForTests } = await import("../src/access.js");
    resetVerifierForTests();
    const nowSeconds = Math.floor(Date.now() / 1000);
    const request = new Request("https://example.com/", {
      headers: { "cf-access-jwt-assertion": "token" },
    });
    request.headers.set(
      "cf-access-jwt-assertion",
      buildJwt({
        iss: "https://example.cloudflareaccess.com",
        aud: ["aud"],
        exp: nowSeconds + 300,
      }),
    );
    const response = await requireAccess(request, {
      ACCESS_AUD: "aud",
      ACCESS_TEAM_DOMAIN: "example.cloudflareaccess.com",
      REQUIRE_ACCESS_VALIDATION: "true",
    });
    expect(response).toBeNull();
  });

  it("returns 401 when the token is missing", async () => {
    const { requireAccess, resetVerifierForTests } = await import("../src/access.js");
    resetVerifierForTests();
    const response = await requireAccess(new Request("https://example.com/"), {
      ACCESS_AUD: "aud",
      ACCESS_TEAM_DOMAIN: "example.cloudflareaccess.com",
      REQUIRE_ACCESS_VALIDATION: "true",
    });
    expect(response.status).toBe(401);
  });
});

function buildJwt(payload) {
  const header = { alg: "RS256", kid: "kid-1" };
  return [header, payload, "signature"]
    .map((part) => encodeBase64Url(typeof part === "string" ? part : JSON.stringify(part)))
    .join(".");
}

function encodeBase64Url(value) {
  return Buffer.from(value, "utf-8")
    .toString("base64")
    .replaceAll("+", "-")
    .replaceAll("/", "_")
    .replaceAll("=", "");
}
