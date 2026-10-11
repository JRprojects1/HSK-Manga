/**
 * Account authentication: users + sessions.
 *
 * Auth model:
 *   - Users are identified by an internal `id` (uuid). Two identity providers:
 *       1. Google OAuth (identifier: google `sub` claim)
 *       2. Magic-link email codes (identifier: verified email)
 *     Both flows converge on the same User record — a user who signs in via
 *     Google gets the email from their Google profile; a user who signs in
 *     via magic-link gets a record keyed to their email. If someone signs in
 *     via Google first and then via magic-link with the same email, we link
 *     them to the same User.
 *
 *   - Sessions are opaque random tokens stored in an HttpOnly `session` cookie.
 *     Server-side we map token → userId with a per-token expiry. Sign-out
 *     deletes the server-side entry (not just the cookie) so a stolen cookie
 *     can be revoked.
 *
 * Storage: JSON files under data/. Small scale (<100 users) doesn't need a
 * real database. Loaded lazily and rewritten on every mutation — sufficient
 * for our write volume (a few writes per user per session).
 */
import { randomBytes, randomInt } from 'crypto';
import * as fs from 'fs';
import * as path from 'path';
import { sendVerificationCode } from './emailService';

// ---------------------------------------------------------------------------
// Data model
// ---------------------------------------------------------------------------

export interface User {
  id: string;
  email: string;
  authMethods: Array<'google' | 'magic-link'>;
  googleId?: string;      // Google `sub` claim, present if the user has ever signed in via Google
  createdAt: number;
  lastLoginAt: number;
  // Sign-in timestamps, oldest first, capped. lastLoginAt alone is overwritten
  // on every login, which made retention ("did they come back?") unmeasurable —
  // this keeps the history. Absent on users created before the field existed.
  loginHistory?: number[];
}

// Enough for ~2 logins/day for 3 months; beyond that the oldest are dropped.
const LOGIN_HISTORY_MAX = 200;

/** Record a sign-in: bumps lastLoginAt and appends to loginHistory (capped). */
function markLogin(user: User, now: number = Date.now()): void {
  user.lastLoginAt = now;
  if (!user.loginHistory) user.loginHistory = [];
  user.loginHistory.push(now);
  if (user.loginHistory.length > LOGIN_HISTORY_MAX) {
    user.loginHistory = user.loginHistory.slice(-LOGIN_HISTORY_MAX);
  }
}

interface UsersFile {
  users: Record<string, User>;
  emailToUserId: Record<string, string>;
  googleIdToUserId: Record<string, string>;
}

interface Session {
  userId: string;
  createdAt: number;
  expiresAt: number;
}

interface SessionsFile {
  sessions: Record<string, Session>;
}

const DATA_DIR = path.resolve(process.cwd(), 'data');
const USERS_PATH = path.join(DATA_DIR, 'users.json');
const SESSIONS_PATH = path.join(DATA_DIR, 'sessions.json');
const SESSION_TTL_MS = 30 * 24 * 60 * 60 * 1000; // 30 days
export const SESSION_COOKIE_NAME = 'session';

// ---------------------------------------------------------------------------
// File-backed load/save (JSON, atomic write to avoid corruption on crash)
// ---------------------------------------------------------------------------

function ensureDataDir(): void {
  if (!fs.existsSync(DATA_DIR)) fs.mkdirSync(DATA_DIR, { recursive: true });
}

function loadUsers(): UsersFile {
  try {
    if (!fs.existsSync(USERS_PATH)) return { users: {}, emailToUserId: {}, googleIdToUserId: {} };
    const raw = fs.readFileSync(USERS_PATH, 'utf-8');
    const parsed = JSON.parse(raw);
    return {
      users: parsed.users || {},
      emailToUserId: parsed.emailToUserId || {},
      googleIdToUserId: parsed.googleIdToUserId || {},
    };
  } catch (e) {
    console.error('[authService] Failed to load users.json, using empty state:', e);
    return { users: {}, emailToUserId: {}, googleIdToUserId: {} };
  }
}

function saveUsers(state: UsersFile): void {
  ensureDataDir();
  const tmp = USERS_PATH + '.tmp';
  fs.writeFileSync(tmp, JSON.stringify(state, null, 2));
  fs.renameSync(tmp, USERS_PATH);
}

function loadSessions(): SessionsFile {
  try {
    if (!fs.existsSync(SESSIONS_PATH)) return { sessions: {} };
    const raw = fs.readFileSync(SESSIONS_PATH, 'utf-8');
    return { sessions: JSON.parse(raw).sessions || {} };
  } catch (e) {
    console.error('[authService] Failed to load sessions.json, using empty state:', e);
    return { sessions: {} };
  }
}

function saveSessions(state: SessionsFile): void {
  ensureDataDir();
  const tmp = SESSIONS_PATH + '.tmp';
  fs.writeFileSync(tmp, JSON.stringify(state, null, 2));
  fs.renameSync(tmp, SESSIONS_PATH);
}

// In-memory copy, refreshed on mutation. Reads are hot; writes are rare.
let usersState: UsersFile = loadUsers();
let sessionsState: SessionsFile = loadSessions();

// ---------------------------------------------------------------------------
// Users
// ---------------------------------------------------------------------------

function normEmail(email: string): string {
  return email.trim().toLowerCase();
}

function newUserId(): string {
  return 'u_' + randomBytes(10).toString('hex');
}

/**
 * Look up a user by email (case-insensitive), returning null if not found.
 */
export function getUserByEmail(email: string): User | null {
  const e = normEmail(email);
  const id = usersState.emailToUserId[e];
  return id ? usersState.users[id] || null : null;
}

/**
 * Look up a user by Google `sub` id.
 */
export function getUserByGoogleId(googleId: string): User | null {
  const id = usersState.googleIdToUserId[googleId];
  return id ? usersState.users[id] || null : null;
}

/**
 * Get a user by id. Returns null if the id doesn't match anyone.
 */
export function getUserById(id: string): User | null {
  return usersState.users[id] || null;
}

/**
 * Sign-in via Google. If a user with this googleId exists, updates lastLogin
 * and returns them. Otherwise creates a new user, or links to an existing
 * email-based user if the email matches.
 */
export function upsertUserFromGoogle(googleId: string, email: string): User {
  const e = normEmail(email);
  // Case 1: we've seen this Google id before.
  const byGoogle = getUserByGoogleId(googleId);
  if (byGoogle) {
    markLogin(byGoogle);
    if (!byGoogle.authMethods.includes('google')) byGoogle.authMethods.push('google');
    saveUsers(usersState);
    return byGoogle;
  }
  // Case 2: user exists by email (magic-link signup earlier). Link Google to it.
  const byEmail = getUserByEmail(e);
  if (byEmail) {
    byEmail.googleId = googleId;
    markLogin(byEmail);
    if (!byEmail.authMethods.includes('google')) byEmail.authMethods.push('google');
    usersState.googleIdToUserId[googleId] = byEmail.id;
    saveUsers(usersState);
    return byEmail;
  }
  // Case 3: brand-new user.
  const id = newUserId();
  const now = Date.now();
  const user: User = {
    id, email: e, authMethods: ['google'], googleId,
    createdAt: now, lastLoginAt: now, loginHistory: [now],
  };
  usersState.users[id] = user;
  usersState.emailToUserId[e] = id;
  usersState.googleIdToUserId[googleId] = id;
  saveUsers(usersState);
  return user;
}

/**
 * Sign-in via magic-link. If the email already has a user, updates lastLogin
 * and returns them. Otherwise creates a new user.
 */
export function upsertUserFromMagicLink(email: string): User {
  const e = normEmail(email);
  const existing = getUserByEmail(e);
  if (existing) {
    markLogin(existing);
    if (!existing.authMethods.includes('magic-link')) existing.authMethods.push('magic-link');
    saveUsers(usersState);
    return existing;
  }
  const id = newUserId();
  const now = Date.now();
  const user: User = {
    id, email: e, authMethods: ['magic-link'],
    createdAt: now, lastLoginAt: now, loginHistory: [now],
  };
  usersState.users[id] = user;
  usersState.emailToUserId[e] = id;
  saveUsers(usersState);
  return user;
}

// ---------------------------------------------------------------------------
// Sessions
// ---------------------------------------------------------------------------

function newSessionToken(): string {
  return randomBytes(32).toString('hex'); // 256 bits of entropy
}

/**
 * Create a new session for `userId`. Returns the token to set as the cookie
 * value. Older sessions for the same user are NOT invalidated — a user can
 * be signed in on multiple devices at once.
 */
export function createSession(userId: string): { token: string; expiresAt: number } {
  const token = newSessionToken();
  const now = Date.now();
  const expiresAt = now + SESSION_TTL_MS;
  sessionsState.sessions[token] = { userId, createdAt: now, expiresAt };
  saveSessions(sessionsState);
  return { token, expiresAt };
}

/**
 * Resolve a session token to its user. Returns null if the token is unknown
 * or expired. Expired sessions are lazily cleaned up on lookup.
 */
export function getUserBySessionToken(token: string): User | null {
  if (!token) return null;
  const session = sessionsState.sessions[token];
  if (!session) return null;
  if (Date.now() > session.expiresAt) {
    delete sessionsState.sessions[token];
    saveSessions(sessionsState);
    return null;
  }
  return getUserById(session.userId);
}

/**
 * Invalidate a session (sign-out). Idempotent — no error if the token
 * doesn't exist or has already expired.
 */
export function deleteSession(token: string): void {
  if (!token) return;
  if (sessionsState.sessions[token]) {
    delete sessionsState.sessions[token];
    saveSessions(sessionsState);
  }
}

/**
 * Sweep expired sessions from disk. Called periodically so the file doesn't
 * grow unbounded from abandoned sessions. Cheap: it's a linear scan over
 * an in-memory object.
 */
export function pruneExpiredSessions(): void {
  const now = Date.now();
  let removed = 0;
  for (const [token, session] of Object.entries(sessionsState.sessions)) {
    if (session.expiresAt < now) {
      delete sessionsState.sessions[token];
      removed++;
    }
  }
  if (removed > 0) saveSessions(sessionsState);
}

// Prune every 6 hours. Runs in-process so it dies with the server; that's fine
// since it'll re-run on next startup.
setInterval(pruneExpiredSessions, 6 * 60 * 60 * 1000).unref();

// ---------------------------------------------------------------------------
// Magic-link codes (account auth, separate from the credit-restore flow)
// ---------------------------------------------------------------------------
//
// Similar rate-limiting model to emailVerifyService but ALWAYS sends the code
// regardless of whether the email has an existing user — magic-link is
// intended for sign-up as well as sign-in, so we can't reveal account
// existence to the caller.

interface PendingCode {
  code: string;
  expiresAt: number;
  attempts: number;
  lastSentAt: number;
}

const AUTH_CODE_TTL_MS = 15 * 60 * 1000;
const AUTH_MAX_VERIFY_ATTEMPTS = 5;
const AUTH_RESEND_COOLDOWN_MS = 60 * 1000;
const AUTH_EMAIL_MAX_PER_HOUR = 5;
const AUTH_IP_MAX_PER_HOUR = 15;
const HOUR_MS = 60 * 60 * 1000;

const pendingCodes = new Map<string, PendingCode>();
const authEmailHistory = new Map<string, number[]>();
const authIpHistory = new Map<string, number[]>();

function withinWindow(list: number[] | undefined, windowMs: number): number[] {
  const cutoff = Date.now() - windowMs;
  return (list || []).filter((t) => t >= cutoff);
}

export function isValidEmail(email: unknown): email is string {
  return (
    typeof email === 'string' &&
    email.trim().length <= 254 &&
    /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email.trim())
  );
}

/**
 * Request a magic-link code for account auth. Rate-limited per email and per
 * IP. Always sends the code (unlike the credit-restore flow which suppresses
 * unknown emails) — for account signup we can't reveal whether the email
 * already has an account, so we can't ALSO reveal whether it doesn't.
 */
export async function requestMagicLinkCode(
  email: string,
  ip: string,
): Promise<{ ok: boolean; reason?: string }> {
  if (!isValidEmail(email)) return { ok: false, reason: 'Please enter a valid email address.' };
  const e = normEmail(email);

  const ipReq = withinWindow(authIpHistory.get(ip), HOUR_MS);
  if (ipReq.length >= AUTH_IP_MAX_PER_HOUR) {
    return { ok: false, reason: 'Too many requests. Please try again later.' };
  }
  const emailReq = withinWindow(authEmailHistory.get(e), HOUR_MS);
  if (emailReq.length >= AUTH_EMAIL_MAX_PER_HOUR) {
    return { ok: false, reason: 'Too many code requests for this email. Try again later.' };
  }
  authIpHistory.set(ip, [...ipReq, Date.now()]);
  authEmailHistory.set(e, [...emailReq, Date.now()]);

  const existing = pendingCodes.get(e);
  const freshEnough =
    existing && Date.now() < existing.expiresAt &&
    Date.now() - existing.lastSentAt < AUTH_RESEND_COOLDOWN_MS;
  if (!freshEnough) {
    const code = String(randomInt(0, 1_000_000)).padStart(6, '0');
    pendingCodes.set(e, {
      code,
      expiresAt: Date.now() + AUTH_CODE_TTL_MS,
      attempts: 0,
      lastSentAt: Date.now(),
    });
    try {
      await sendVerificationCode(e, code);
    } catch (err) {
      console.error('[authService] Failed to send magic-link code:', err);
      return { ok: false, reason: 'Could not send the email right now. Please try again shortly.' };
    }
  }
  return { ok: true };
}

/**
 * Verify a magic-link code. On success the code is consumed (single-use)
 * and the user is upserted from the email. Returns the resolved User.
 */
export function verifyMagicLinkCode(
  email: string,
  code: string,
): { ok: boolean; user?: User; reason?: string } {
  if (!isValidEmail(email)) return { ok: false, reason: 'Invalid email.' };
  if (typeof code !== 'string') return { ok: false, reason: 'Code required.' };
  const e = normEmail(email);
  const entry = pendingCodes.get(e);

  if (!entry) return { ok: false, reason: 'No code was requested, or it has expired. Request a new one.' };
  if (Date.now() > entry.expiresAt) {
    pendingCodes.delete(e);
    return { ok: false, reason: 'That code expired. Request a new one.' };
  }
  if (entry.attempts >= AUTH_MAX_VERIFY_ATTEMPTS) {
    pendingCodes.delete(e);
    return { ok: false, reason: 'Too many incorrect attempts. Request a new code.' };
  }
  entry.attempts++;
  if (code.trim() !== entry.code) {
    return { ok: false, reason: 'Incorrect code. Please check and try again.' };
  }
  pendingCodes.delete(e);
  const user = upsertUserFromMagicLink(e);
  return { ok: true, user };
}
