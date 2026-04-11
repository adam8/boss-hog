let cachedJwks = null;

function getRequiredVar(env, name) {
  const value = env[name];
  if (!value || !String(value).trim()) {
    throw new Error(`Missing required UI worker variable: ${name}`);
  }
  return String(value).trim();
}

async function getJwks(env) {
  const teamDomain = getRequiredVar(env, "ACCESS_TEAM_DOMAIN");
  if (!cachedJwks || cachedJwks.teamDomain !== teamDomain) {
    const response = await fetch(`https://${teamDomain}/cdn-cgi/access/certs`);
    if (!response.ok) {
      throw new Error(`Unable to fetch Cloudflare Access certs (${response.status}).`);
    }
    cachedJwks = {
      teamDomain,
      keys: (await response.json()).keys ?? [],
    };
  }
  return cachedJwks.keys;
}

function shouldValidateAccess(env) {
  return String(env.REQUIRE_ACCESS_VALIDATION ?? "true").toLowerCase() !== "false";
}

export async function requireAccess(request, env) {
  if (!shouldValidateAccess(env)) {
    return null;
  }

  const token = request.headers.get("cf-access-jwt-assertion");
  if (!token) {
    return new Response("Missing Cloudflare Access token.", { status: 401 });
  }

  try {
    await verifyAccessJwt(token, env);
    return null;
  } catch (error) {
    return new Response("Invalid Cloudflare Access token.", { status: 403 });
  }
}

export function resetVerifierForTests() {
  cachedJwks = null;
}

async function verifyAccessJwt(token, env) {
  const [encodedHeader, encodedPayload, encodedSignature] = token.split(".");
  if (!encodedHeader || !encodedPayload || !encodedSignature) {
    throw new Error("Malformed JWT.");
  }

  const header = JSON.parse(decodeBase64Url(encodedHeader));
  const payload = JSON.parse(decodeBase64Url(encodedPayload));
  const signature = base64UrlToBytes(encodedSignature);
  const signingInput = new TextEncoder().encode(`${encodedHeader}.${encodedPayload}`);
  const key = await resolveVerificationKey(header, env);
  const verified = await crypto.subtle.verify(
    "RSASSA-PKCS1-v1_5",
    key,
    signature,
    signingInput,
  );

  if (!verified) {
    throw new Error("Signature verification failed.");
  }
  validateClaims(payload, env);
}

async function resolveVerificationKey(header, env) {
  const keys = await getJwks(env);
  const matchingKey = keys.find((key) => key.kid === header.kid) ?? keys[0];
  if (!matchingKey) {
    throw new Error("No matching signing key found.");
  }
  return crypto.subtle.importKey(
    "jwk",
    matchingKey,
    {
      name: "RSASSA-PKCS1-v1_5",
      hash: "SHA-256",
    },
    false,
    ["verify"],
  );
}

function validateClaims(payload, env) {
  const issuer = `https://${getRequiredVar(env, "ACCESS_TEAM_DOMAIN")}`;
  const audience = getRequiredVar(env, "ACCESS_AUD");
  const nowSeconds = Math.floor(Date.now() / 1000);

  if (payload.iss !== issuer) {
    throw new Error("Unexpected issuer.");
  }

  const payloadAudience = Array.isArray(payload.aud) ? payload.aud : [payload.aud];
  if (!payloadAudience.includes(audience)) {
    throw new Error("Unexpected audience.");
  }

  if (typeof payload.exp === "number" && payload.exp < nowSeconds) {
    throw new Error("Token expired.");
  }

  if (typeof payload.nbf === "number" && payload.nbf > nowSeconds) {
    throw new Error("Token not active yet.");
  }
}

function decodeBase64Url(value) {
  const base64 = value.replaceAll("-", "+").replaceAll("_", "/").padEnd(Math.ceil(value.length / 4) * 4, "=");
  if (typeof atob === "function") {
    return atob(base64);
  }
  return Buffer.from(base64, "base64").toString("utf-8");
}

function base64UrlToBytes(value) {
  const decoded = decodeBase64Url(value);
  return Uint8Array.from(decoded, (character) => character.charCodeAt(0));
}
