/**
 * Authentication routes: Google OAuth + magic-link + session management.
 *
 * Endpoints:
 *   GET  /api/auth/google              — redirect to Google consent screen
 *   GET  /api/auth/google/callback     — handle OAuth callback, set session
 *   POST /api/auth/magic-link/request  — email a 6-digit sign-in code
 *   POST /api/auth/magic-link/verify   — verify the code, set session
 *   GET  /api/me                       — return current user (or null)
 *   POST /api/auth/logout              — clear session
 *
 * Session cookie: `session=<64-char hex>`, HttpOnly, Secure, SameSite=Lax
 * (Lax so the Google OAuth callback redirect can carry it), 30-day expiry.
 * Server-side sessions in data/sessions.json so we can revoke.
 */
import { Router, Request, Response } from 'express';
import { randomBytes } from 'crypto';
import * as https from 'https';
import * as querystring from 'querystring';
import {
  SESSION_COOKIE_NAME,
  createSession,
  deleteSession,
  getUserBySessionToken,
  upsertUserFromGoogle,
  requestMagicLinkCode,
  verifyMagicLinkCode,
  isValidEmail,
  User,
} from '../services/authService';
// NOTE: Sign-in intentionally does NOT auto-import a browser's chapter history
// into the account. The old linkGuestHistoryOnSignIn sweep linked the browser
// to whoever signed in and pulled that browser's whole submission history into
// their library — which leaked one account's chapters into the next account
// that signed in on the same browser. Anonymous chapters now enter an account
// ONLY via the explicit /api/library/claim flow (user opts in).

const router = Router();

const GOOGLE_CLIENT_ID = process.env.GOOGLE_CLIENT_ID;
const GOOGLE_CLIENT_SECRET = process.env.GOOGLE_CLIENT_SECRET;
const GOOGLE_REDIRECT_URI_DEV = process.env.GOOGLE_REDIRECT_URI_DEV;
const GOOGLE_REDIRECT_URI_PROD = process.env.GOOGLE_REDIRECT_URI_PROD;
// Legacy single-URI env var, kept as a fallback so old .env files still work.
const GOOGLE_REDIRECT_URI_LEGACY = process.env.GOOGLE_REDIRECT_URI;
const SESSION_TTL_MS = 30 * 24 * 60 * 60 * 1000;

function getClientIp(req: Request): string {
  const cfIp = req.headers['cf-connecting-ip'];
  if (typeof cfIp === 'string' && cfIp) return cfIp;
  const xff = req.headers['x-forwarded-for'];
  if (typeof xff === 'string' && xff) return xff.split(',')[0].trim();
  return req.ip || 'unknown';
}

/**
 * Pick the right OAuth redirect URI for the incoming request. Dev vs prod
 * is determined by the Host header — localhost/127.0.0.1 gets the dev URI,
 * anything else gets prod. This means the SAME .env can be committed on
 * both machines (Computer 1 and Computer 2) without swapping.
 *
 * Falls back to the legacy GOOGLE_REDIRECT_URI env var if the newer pair
 * isn't set, so existing deployments keep working after a partial update.
 */
function getGoogleRedirectUri(req: Request): string | null {
  const host = (req.get('host') || '').toLowerCase();
  const isLocal = host.startsWith('localhost') || host.startsWith('127.0.0.1');
  const preferred = isLocal ? GOOGLE_REDIRECT_URI_DEV : GOOGLE_REDIRECT_URI_PROD;
  return preferred || GOOGLE_REDIRECT_URI_LEGACY || null;
}

function setSessionCookie(res: Response, token: string, expiresAt: number): void {
  res.cookie(SESSION_COOKIE_NAME, token, {
    httpOnly: true,
    secure: true,
    sameSite: 'lax',
    maxAge: expiresAt - Date.now(),
    path: '/',
  });
}

function clearSessionCookie(res: Response): void {
  res.clearCookie(SESSION_COOKIE_NAME, { path: '/' });
}

// Public shape returned to the frontend — omits internal audit fields.
function publicUser(u: User) {
  return {
    id: u.id,
    email: u.email,
    authMethods: u.authMethods,
    createdAt: u.createdAt,
  };
}

// ---------------------------------------------------------------------------
// State-parameter CSRF protection for the Google OAuth flow. We generate a
// random token, stash it in a short-lived cookie, and require it to come back
// unmodified in the callback query string. Prevents an attacker from tricking
// a signed-in user into linking a different Google account.
// ---------------------------------------------------------------------------

const STATE_COOKIE_NAME = 'oauth_state';
const RETURN_COOKIE_NAME = 'oauth_return';
const STATE_TTL_MS = 10 * 60 * 1000;

// Only allow a same-origin app path as the post-sign-in return target (prevents
// an open-redirect via the ?return= param). Defaults to /start.
function sanitizeReturnPath(v: unknown): string {
  if (typeof v !== 'string') return '/start';
  if (!v.startsWith('/') || v.startsWith('//') || v.includes('://')) return '/start';
  if (v.startsWith('/api/')) return '/start';
  return v;
}

// ---------------------------------------------------------------------------
// GET /api/auth/google — redirect to Google
// ---------------------------------------------------------------------------

router.get('/auth/google', (req: Request, res: Response) => {
  const redirectUri = getGoogleRedirectUri(req);
  if (!GOOGLE_CLIENT_ID || !redirectUri) {
    res.status(503).send('Google OAuth is not configured on this server.');
    return;
  }
  const state = randomBytes(24).toString('hex');
  res.cookie(STATE_COOKIE_NAME, state, {
    httpOnly: true,
    secure: true,
    sameSite: 'lax',
    maxAge: STATE_TTL_MS,
    path: '/',
  });
  // Remember where the user started so the callback can return them there.
  res.cookie(RETURN_COOKIE_NAME, sanitizeReturnPath(req.query.return), {
    httpOnly: true, secure: true, sameSite: 'lax', maxAge: STATE_TTL_MS, path: '/',
  });
  const params = new URLSearchParams({
    client_id: GOOGLE_CLIENT_ID,
    redirect_uri: redirectUri,
    response_type: 'code',
    scope: 'openid email profile',
    state,
    // access_type=online — we don't need refresh tokens for a sign-in-only
    // integration. Simplifies the flow and side-steps refresh-token storage.
    access_type: 'online',
    prompt: 'select_account',
  });
  res.redirect(`https://accounts.google.com/o/oauth2/v2/auth?${params.toString()}`);
});

// ---------------------------------------------------------------------------
// GET /api/auth/google/callback — exchange code for tokens, create session
// ---------------------------------------------------------------------------

interface GoogleTokenResponse {
  access_token?: string;
  id_token?: string;
  error?: string;
  error_description?: string;
}

/** POST to Google's token endpoint. Returns the parsed JSON body. */
function exchangeGoogleCode(code: string, redirectUri: string): Promise<GoogleTokenResponse> {
  return new Promise((resolve, reject) => {
    const body = querystring.stringify({
      code,
      client_id: GOOGLE_CLIENT_ID!,
      client_secret: GOOGLE_CLIENT_SECRET!,
      redirect_uri: redirectUri,
      grant_type: 'authorization_code',
    });
    const req = https.request(
      {
        method: 'POST',
        hostname: 'oauth2.googleapis.com',
        path: '/token',
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded',
          'Content-Length': Buffer.byteLength(body),
        },
      },
      (r) => {
        let data = '';
        r.on('data', (c) => (data += c));
        r.on('end', () => {
          try { resolve(JSON.parse(data)); }
          catch (e) { reject(new Error('Google token endpoint returned non-JSON: ' + data.slice(0, 200))); }
        });
      },
    );
    req.on('error', reject);
    req.setTimeout(10000, () => { req.destroy(new Error('Google token exchange timeout')); });
    req.write(body);
    req.end();
  });
}

/**
 * Parse Google's id_token JWT. We don't verify the signature here because
 * we ONLY trust id_tokens that came directly from the token endpoint (server-
 * to-server over TLS with our client secret), never from the browser. This
 * side-steps the whole JWK / public-key rotation dance.
 */
function decodeIdToken(idToken: string): { sub?: string; email?: string; email_verified?: boolean; name?: string } {
  const parts = idToken.split('.');
  if (parts.length !== 3) throw new Error('Malformed id_token');
  const payload = Buffer.from(parts[1].replace(/-/g, '+').replace(/_/g, '/'), 'base64').toString('utf-8');
  return JSON.parse(payload);
}

router.get('/auth/google/callback', async (req: Request, res: Response) => {
  const redirectUri = getGoogleRedirectUri(req);
  if (!GOOGLE_CLIENT_ID || !GOOGLE_CLIENT_SECRET || !redirectUri) {
    res.status(503).send('Google OAuth is not configured on this server.');
    return;
  }
  const code = String(req.query.code || '');
  const state = String(req.query.state || '');
  const stateCookie = (req.cookies || {})[STATE_COOKIE_NAME];
  res.clearCookie(STATE_COOKIE_NAME, { path: '/' });

  if (!code) { res.status(400).send('Missing authorization code from Google.'); return; }
  if (!state || state !== stateCookie) {
    // CSRF defense: the state we generated didn't come back. Refuse.
    res.status(400).send('OAuth state mismatch. Please try signing in again.');
    return;
  }

  let tokenResp: GoogleTokenResponse;
  try { tokenResp = await exchangeGoogleCode(code, redirectUri); }
  catch (e: any) {
    console.error('[auth-google] Token exchange failed:', e.message);
    res.status(502).send('Could not reach Google to complete sign-in. Please try again.');
    return;
  }
  if (tokenResp.error || !tokenResp.id_token) {
    console.error('[auth-google] Google returned error:', tokenResp);
    res.status(400).send('Google declined the sign-in: ' + (tokenResp.error_description || tokenResp.error || 'unknown error'));
    return;
  }

  let claims: ReturnType<typeof decodeIdToken>;
  try { claims = decodeIdToken(tokenResp.id_token); }
  catch (e: any) {
    console.error('[auth-google] id_token decode failed:', e.message);
    res.status(400).send('Google returned an id_token we could not read.');
    return;
  }

  if (!claims.sub || !claims.email || !claims.email_verified) {
    // Only accept verified emails. An unverified Google email is rare (Google
    // requires verification for account creation) but if it happens, refuse.
    res.status(400).send('Your Google account does not have a verified email.');
    return;
  }

  const user = upsertUserFromGoogle(claims.sub, claims.email);
  const { token, expiresAt } = createSession(user.id);
  setSessionCookie(res, token, expiresAt);
  console.log('[auth-google] Signed in user:', user.email);
  // Return to the page the user started from (validated same-origin path,
  // default /start) with a one-time flag so the app can detect a FRESH sign-in
  // and surface the claim step for any chapters saved in the browser.
  const returnTo = sanitizeReturnPath((req.cookies || {}).oauth_return);
  res.clearCookie(RETURN_COOKIE_NAME, { path: '/' });
  res.redirect(returnTo + (returnTo.indexOf('?') >= 0 ? '&' : '?') + 'signedin=1');
});

// ---------------------------------------------------------------------------
// Magic-link (email code) — sign-up and sign-in in one flow
// ---------------------------------------------------------------------------

router.post('/auth/magic-link/request', async (req: Request, res: Response) => {
  const email = req.body?.email;
  if (!isValidEmail(email)) {
    res.status(400).json({ code: 'BAD_EMAIL', error: 'Please enter a valid email address.' });
    return;
  }
  const result = await requestMagicLinkCode(email, getClientIp(req));
  if (!result.ok) {
    res.status(429).json({ code: 'RATE_LIMITED', error: result.reason });
    return;
  }
  res.json({ ok: true });
});

router.post('/auth/magic-link/verify', (req: Request, res: Response) => {
  const email = req.body?.email;
  const code = req.body?.code;
  const result = verifyMagicLinkCode(email, code);
  if (!result.ok || !result.user) {
    res.status(400).json({ code: 'BAD_CODE', error: result.reason || 'Invalid code.' });
    return;
  }
  const { token, expiresAt } = createSession(result.user.id);
  setSessionCookie(res, token, expiresAt);
  res.json({ ok: true, user: publicUser(result.user) });
});

// ---------------------------------------------------------------------------
// GET /api/me — return the currently signed-in user (or null for guests)
// ---------------------------------------------------------------------------

router.get('/me', (req: Request, res: Response) => {
  const token = (req.cookies || {})[SESSION_COOKIE_NAME];
  const user = token ? getUserBySessionToken(token) : null;
  res.json({ user: user ? publicUser(user) : null });
});

// ---------------------------------------------------------------------------
// POST /api/auth/logout — clear session cookie + revoke server-side entry
// ---------------------------------------------------------------------------

router.post('/auth/logout', (req: Request, res: Response) => {
  const token = (req.cookies || {})[SESSION_COOKIE_NAME];
  if (token) deleteSession(token);
  clearSessionCookie(res);
  res.json({ ok: true });
});

export default router;
