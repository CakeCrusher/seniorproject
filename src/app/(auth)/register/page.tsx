"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";

export default function RegisterPage() {
  const router = useRouter();
  const [form, setForm] = useState({ username: "", password: "" });
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  function updateField(field: string, value: string) {
    setForm((prev) => ({ ...prev, [field]: value }));
  }

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError("");
    setBusy(true);

    try {
      const res = await fetch("/api/auth/register", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(form),
      });
      const data = await res.json();

      if (!res.ok) {
        setError(data.error);
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
    <div className="min-h-screen flex items-center justify-center px-4">
      <div className="w-full max-w-sm">
        <h1 className="text-2xl font-bold text-center mb-8">Create Account</h1>
        <form onSubmit={onSubmit} className="space-y-4">
          {error && (
            <div className="bg-red-900/50 border border-red-700 text-red-200 px-4 py-2 rounded text-sm">
              {error}
            </div>
          )}
          <div>
            <label className="block text-sm text-gray-400 mb-1">Username</label>
            <input
              type="text"
              value={form.username}
              onChange={(e) => updateField("username", e.target.value)}
              className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded focus:outline-none focus:border-blue-500"
              required
              minLength={3}
            />
          </div>
          <div>
            <label className="block text-sm text-gray-400 mb-1">Password</label>
            <input
              type="password"
              value={form.password}
              onChange={(e) => updateField("password", e.target.value)}
              className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded focus:outline-none focus:border-blue-500"
              required
              minLength={6}
            />
          </div>
          <button
            type="submit"
            disabled={busy}
            className="w-full py-2 bg-blue-600 hover:bg-blue-700 disabled:opacity-50 rounded font-medium transition"
          >
            {busy ? "Hold on..." : "Create Account"}
          </button>
        </form>
        <p className="text-center text-sm text-gray-500 mt-4">
          Already registered?{" "}
          <Link href="/login" className="text-blue-400 hover:underline">
            Sign in
          </Link>
        </p>
      </div>
    </div>
  );
}
