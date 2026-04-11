import { requireAccess } from "./access.js";

function jsonError(message, status) {
  return new Response(JSON.stringify({ error: message }), {
    status,
    headers: { "content-type": "application/json; charset=utf-8" },
  });
}

function buildApiRequest(request) {
  const url = new URL(request.url);
  return new Request(`https://hog-api.internal${url.pathname.replace(/^\/api/, "")}${url.search}`, {
    method: "GET",
    headers: {
      "accept": "application/json",
    },
  });
}

async function serveIndex(request, env) {
  const accessFailure = await requireAccess(request, env);
  if (accessFailure) {
    return accessFailure;
  }
  const indexUrl = new URL(request.url);
  indexUrl.pathname = "/app/index.html";
  indexUrl.search = "";
  return env.ASSETS.fetch(indexUrl.toString());
}

async function handleApi(request, env) {
  const accessFailure = await requireAccess(request, env);
  if (accessFailure) {
    return accessFailure;
  }
  try {
    const upstreamResponse = await env.HOG_API.fetch(buildApiRequest(request));
    return new Response(upstreamResponse.body, {
      status: upstreamResponse.status,
      headers: {
        "content-type": upstreamResponse.headers.get("content-type") || "application/json; charset=utf-8",
        "cache-control": "no-store",
      },
    });
  } catch (error) {
    console.error(JSON.stringify({ event: "api_proxy_error", message: String(error) }));
    return jsonError("Backtest service unavailable.", 502);
  }
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    if (url.pathname === "/" || url.pathname === "/index.html") {
      return serveIndex(request, env);
    }
    if (url.pathname.startsWith("/api/")) {
      return handleApi(request, env);
    }
    return env.ASSETS.fetch(request);
  },
};
