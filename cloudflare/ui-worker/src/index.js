import { APP_CSS, APP_JS, INDEX_HTML, LOGO_PNG_BASE64 } from "./site.generated.js";
import { requireAccess } from "./access.js";

let logoBytes;

function jsonError(message, status) {
  return new Response(JSON.stringify({ error: message }), {
    status,
    headers: { "content-type": "application/json; charset=utf-8" },
  });
}

function textResponse(body, contentType) {
  return new Response(body, {
    status: 200,
    headers: {
      "content-type": contentType,
      "cache-control": "no-store",
    },
  });
}

function decodeBase64(base64) {
  const binary = atob(base64);
  return Uint8Array.from(binary, (char) => char.charCodeAt(0));
}

function imageResponse(base64, contentType) {
  logoBytes ??= decodeBase64(base64);
  return new Response(logoBytes, {
    status: 200,
    headers: {
      "content-type": contentType,
      "cache-control": "public, max-age=3600",
    },
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
  return textResponse(INDEX_HTML, "text/html; charset=utf-8");
}

async function serveStaticAsset(request, env, body, contentType) {
  const accessFailure = await requireAccess(request, env);
  if (accessFailure) {
    return accessFailure;
  }
  return textResponse(body, contentType);
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
    if (url.pathname === "/app/styles.css") {
      return serveStaticAsset(request, env, APP_CSS, "text/css; charset=utf-8");
    }
    if (url.pathname === "/app/app.js") {
      return serveStaticAsset(request, env, APP_JS, "text/javascript; charset=utf-8");
    }
    if (url.pathname === "/app/logo.png") {
      const accessFailure = await requireAccess(request, env);
      if (accessFailure) {
        return accessFailure;
      }
      return imageResponse(LOGO_PNG_BASE64, "image/png");
    }
    if (url.pathname.startsWith("/api/")) {
      return handleApi(request, env);
    }
    return new Response("Not found.", { status: 404 });
  },
};
