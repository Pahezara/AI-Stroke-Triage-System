import { useState } from "react";

export default function Login({ apiBase, onLogin }) {
    const [username, setUsername] = useState("admin");
    const [password, setPassword] = useState("");
    const [err, setErr] = useState("");
    const [loading, setLoading] = useState(false);

    async function handleLogin(e) {
        e.preventDefault();
        setErr("");
        setLoading(true);

        try {
            const res = await fetch(`${apiBase}/auth/login`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ username, password }),
            });

            if (!res.ok) {
                const detail = await res.json().catch(() => ({}));
                throw new Error(detail.detail || `Login failed: HTTP ${res.status}`);
            }

            const data = await res.json();
            onLogin(data.access_token, username);
        } catch (e2) {
            setErr(e2.message || "Login failed");
        } finally {
            setLoading(false);
        }
    }

    return (
        <div className="min-h-screen bg-slate-950 text-slate-100 flex items-center justify-center px-4">
            <div className="w-full max-w-md rounded-3xl border border-slate-800 bg-slate-900/40 p-6 shadow-2xl">
                <div className="mb-6 text-center">
                    <div className="text-4xl mb-2">🧠</div>
                    <h1 className="text-2xl font-extrabold">Stroke AI Triage</h1>
                    <p className="mt-1 text-sm text-slate-400">
                        Admin access required
                    </p>
                </div>

                <form onSubmit={handleLogin} className="space-y-4">
                    <div>
                        <label className="text-xs font-semibold text-slate-300">
                            Username
                        </label>
                        <input
                            className="mt-1 w-full rounded-xl border border-slate-700 bg-slate-950/60 px-3 py-2 text-sm outline-none focus:border-sky-500"
                            value={username}
                            onChange={(e) => setUsername(e.target.value)}
                            placeholder="admin"
                            autoComplete="username"
                        />
                    </div>

                    <div>
                        <label className="text-xs font-semibold text-slate-300">
                            Password
                        </label>
                        <input
                            className="mt-1 w-full rounded-xl border border-slate-700 bg-slate-950/60 px-3 py-2 text-sm outline-none focus:border-sky-500"
                            type="password"
                            value={password}
                            onChange={(e) => setPassword(e.target.value)}
                            placeholder="Enter admin password"
                            autoComplete="current-password"
                        />
                    </div>

                    {err && (
                        <div className="rounded-xl border border-red-500/30 bg-red-500/10 px-3 py-2 text-sm text-red-200">
                            {err}
                        </div>
                    )}

                    <button
                        type="submit"
                        disabled={loading}
                        className="w-full rounded-xl bg-emerald-500 px-4 py-2 text-sm font-extrabold text-slate-950 hover:bg-emerald-400 disabled:bg-slate-700 disabled:text-slate-200"
                    >
                        {loading ? "Signing in..." : "Sign in as Admin"}
                    </button>
                </form>
            </div>
        </div>
    );
}