// Transactional email — currently just the sign-in verification code.
//
// Sends via Resend's HTTP API (https://resend.com) when RESEND_API_KEY is set.
// We call the REST endpoint directly with fetch, so there is no SDK dependency.
//
// When RESEND_API_KEY is unset, the code is logged to the server console
// instead of emailed. That keeps sign-in fully testable in local
// development without an email provider — the same "no-op when unconfigured"
// pattern the codebase uses for Turnstile.
//
// Setup for production: create a Resend account, verify your sending
// domain, then set in .env:
//   RESEND_API_KEY=re_...
//   EMAIL_FROM="HSK Manga <noreply@your-domain.com>"

const RESEND_API_KEY = process.env.RESEND_API_KEY;
const EMAIL_FROM = process.env.EMAIL_FROM || 'HSK Manga <noreply@hskmanga.com>';

export function isEmailConfigured(): boolean {
  return Boolean(RESEND_API_KEY);
}

/**
 * Send a verification code to `email`. Throws if a configured provider fails,
 * so the caller can surface a "couldn't send" message rather than silently
 * leaving the user waiting for an email that will never arrive.
 */
export async function sendVerificationCode(email: string, code: string): Promise<void> {
  const subject = 'Your HSK Manga code';
  const text =
    `Your HSK Manga verification code is ${code}\n\n` +
    `Enter it on the site to sign in. ` +
    `The code expires in 15 minutes.\n\n` +
    `If you didn't request this, you can safely ignore this email.`;

  if (!RESEND_API_KEY) {
    console.log(`[Email] (dev — RESEND_API_KEY unset) verification code for ${email}: ${code}`);
    return;
  }

  const res = await fetch('https://api.resend.com/emails', {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${RESEND_API_KEY}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ from: EMAIL_FROM, to: email, subject, text }),
  });

  if (!res.ok) {
    const detail = await res.text().catch(() => '');
    throw new Error(`Resend API error ${res.status}: ${detail}`);
  }
}
