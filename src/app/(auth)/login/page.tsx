"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";

export default function LoginPage() {
  const router = useRouter();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError("");
    setBusy(true);

    try {
      const res = await fetch("/api/auth/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username, password }),
      });
      const data = await res.json();

      if (!res.ok) {
        setError(data.error || "Something went wrong");
        return;
      }

      localStorage.setItem("token", data.token);
      localStorage.setItem("user", JSON.stringify(data.user));
      router.push("/chat");
    } catch {
      setError("Couldn't reach the server");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="min-h-screen flex items-center justify-center px-4 bg-[#0c0c0f] font-mono">
      <div className="w-full max-w-sm">

        {/* logo */}
        <div className="text-center mb-8">
          <h1
            className="text-4xl text-[#e8d5a8] mb-1"
            style={{ fontFamily: "'Fraunces', Georgia, serif", fontStyle: "italic", fontWeight: 300 }}
          >
            Dialogue
          </h1>
          <p className="text-[10px] text-[#6b6760] tracking-widest uppercase">sign in to continue</p>
        </div>

        {/* card */}
        <div className="bg-[#131316] border border-white/10 rounded-xl px-8 py-8">
          <form onSubmit={onSubmit} className="space-y-5">

            {error && (
              <div className="text-xs text-[#e07070] bg-[#1f0f0f] border-l-2 border-[#e07070] rounded-r px-3 py-2">
                {error}
              </div>
            )}

            <div>
              <label className="block text-[10px] text-[#6b6760] tracking-[0.5px] uppercase mb-1.5">
                username
              </label>
              <input
                type="text"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                className="w-full px-3 py-2 bg-[#1a1a1f] border border-white/10 rounded-md text-[13px] text-[#ede9e3] placeholder-[#6b6760] outline-none focus:border-[#c9aa71] transition-colors font-mono"
                required
              />
            </div>

            <div>
              <label className="block text-[10px] text-[#6b6760] tracking-[0.5px] uppercase mb-1.5">
                password
              </label>
              <input
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="w-full px-3 py-2 bg-[#1a1a1f] border border-white/10 rounded-md text-[13px] text-[#ede9e3] placeholder-[#6b6760] outline-none focus:border-[#c9aa71] transition-colors font-mono"
                required
              />
            </div>

            <button
              type="submit"
              disabled={busy}
              className="w-full py-2.5 bg-[#c9aa71] hover:bg-[#e8d5a8] disabled:opacity-40 rounded-md text-[13px] text-[#0c0c0f] font-medium transition-colors font-mono"
            >
              {busy ? "signing in…" : "sign in"}
            </button>

          </form>
        </div>

        <p className="text-center text-[11px] text-[#6b6760] mt-5">
          don&apos;t have an account?{" "}
          <Link href="/register" className="text-[#c9aa71] hover:text-[#e8d5a8] transition-colors">
            register
          </Link>
        </p>

      </div>
    </div>
  );
}
