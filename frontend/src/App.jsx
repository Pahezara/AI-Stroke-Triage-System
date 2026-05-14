import { useEffect, useMemo, useState } from "react";
import Login from "./Login.jsx";

const API_BASE = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000";

const RSNA_CLASSES = [
  { value: 0, label: "any" },
  { value: 1, label: "epidural" },
  { value: 2, label: "intraparenchymal" },
  { value: 3, label: "intraventricular" },
  { value: 4, label: "subarachnoid" },
  { value: 5, label: "subdural" },
];

function cls(...xs) {
  return xs.filter(Boolean).join(" ");
}

function Badge({ tone = "gray", children }) {
  const map = {
    gray: "bg-slate-700/60 text-slate-200 border-slate-600/50",
    green: "bg-emerald-600/20 text-emerald-200 border-emerald-500/30",
    orange: "bg-orange-600/20 text-orange-200 border-orange-500/30",
    red: "bg-red-600/20 text-red-200 border-red-500/30",
    blue: "bg-blue-600/20 text-blue-200 border-blue-500/30",
    purple: "bg-purple-600/20 text-purple-200 border-purple-500/30",
  };

  return (
    <span
      className={cls(
        "inline-flex items-center gap-1 rounded-full border px-3 py-1 text-xs font-semibold",
        map[tone] || map.gray
      )}
    >
      {children}
    </span>
  );
}

function ProgressBar({ value }) {
  const pct = Math.max(0, Math.min(100, Math.round(Number(value || 0) * 100)));

  return (
    <div className="h-2 w-full overflow-hidden rounded-full bg-slate-700/60">
      <div className="h-full bg-sky-500/80" style={{ width: `${pct}%` }} />
    </div>
  );
}

function copyToClipboard(text) {
  navigator.clipboard.writeText(text).catch(() => { });
}

function safeFileName(s) {
  return String(s || "")
    .replace(/[<>:"/\\|?*\x00-\x1F]/g, "_")
    .slice(0, 80);
}

export default function App() {
  const saved = useMemo(() => {
    try {
      return JSON.parse(localStorage.getItem("stroke_ai_inputs") || "{}");
    } catch {
      return {};
    }
  }, []);

  const [token, setToken] = useState(() => localStorage.getItem("stroke_ai_token") || "");
  const [adminUser, setAdminUser] = useState(() => localStorage.getItem("stroke_ai_admin") || "");

  const [ctDir, setCtDir] = useState(saved.ctDir || "");
  const [mriDwi, setMriDwi] = useState(saved.mriDwi || "");
  const [mriAdc, setMriAdc] = useState(saved.mriAdc || "");
  const [mriFlair, setMriFlair] = useState(saved.mriFlair || "");

  const [patientId, setPatientId] = useState(saved.patientId || "");
  const [studyId, setStudyId] = useState(saved.studyId || "");

  const [gradcamDicom, setGradcamDicom] = useState(saved.gradcamDicom || "");
  const [gradcamClass, setGradcamClass] = useState(saved.gradcamClass ?? 0);
  const [gradcamUrl, setGradcamUrl] = useState("");

  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [err, setErr] = useState("");

  const [gcLoading, setGcLoading] = useState(false);
  const [gcErr, setGcErr] = useState("");

  const [pdfLoading, setPdfLoading] = useState(false);
  const [pdfErr, setPdfErr] = useState("");

  const [apiUp, setApiUp] = useState(false);

  useEffect(() => {
    localStorage.setItem(
      "stroke_ai_inputs",
      JSON.stringify({
        ctDir,
        mriDwi,
        mriAdc,
        mriFlair,
        patientId,
        studyId,
        gradcamDicom,
        gradcamClass,
      })
    );
  }, [ctDir, mriDwi, mriAdc, mriFlair, patientId, studyId, gradcamDicom, gradcamClass]);

  useEffect(() => {
    let alive = true;

    async function ping() {
      try {
        const r = await fetch(`${API_BASE}/health`, { method: "GET" });
        if (!alive) return;
        setApiUp(r.ok);
      } catch {
        if (!alive) return;
        setApiUp(false);
      }
    }

    ping();
    const t = setInterval(ping, 3000);

    return () => {
      alive = false;
      clearInterval(t);
    };
  }, []);

  function authHeaders(extra = {}) {
    return token ? { ...extra, Authorization: `Bearer ${token}` } : extra;
  }

  function handleLogout() {
    localStorage.removeItem("stroke_ai_token");
    localStorage.removeItem("stroke_ai_admin");
    setToken("");
    setAdminUser("");
    setResult(null);
    setErr("");
    setPdfErr("");
    setGcErr("");
    setGradcamUrl("");
  }

  function handleUnauthorized(status) {
    if (status === 401 || status === 403) {
      handleLogout();
      throw new Error("Session expired or unauthorized. Please login again.");
    }
  }

  const severityTone = (sev) => {
    if (sev === "high") return "red";
    if (sev === "moderate") return "orange";
    if (sev === "low") return "green";
    return "gray";
  };

  const reviewTone = (rp) => {
    if (rp === "urgent") return "red";
    if (rp === "needs_radiologist_confirmation") return "orange";
    return "blue";
  };

  async function runTriage(e) {
    e.preventDefault();
    setErr("");
    setPdfErr("");
    setResult(null);
    setGradcamUrl("");
    setGcErr("");

    setLoading(true);

    try {
      const payload = {
        ct_dir: ctDir || null,
        mri_dwi: mriDwi || null,
        mri_adc: mriAdc || null,
        mri_flair: mriFlair || null,
      };

      const res = await fetch(`${API_BASE}/triage/paths`, {
        method: "POST",
        headers: authHeaders({ "Content-Type": "application/json" }),
        body: JSON.stringify(payload),
      });

      handleUnauthorized(res.status);

      if (!res.ok) {
        const detail = await res.json().catch(() => ({}));
        throw new Error(detail.detail || `HTTP ${res.status}`);
      }

      const data = await res.json();
      setResult(data);
    } catch (e2) {
      setErr(e2?.message || "Triage failed");
    } finally {
      setLoading(false);
    }
  }

  async function runGradcam(e) {
    e.preventDefault();
    setGcErr("");
    setGradcamUrl("");

    if (!gradcamDicom) {
      setGcErr("Enter a CT DICOM slice path.");
      return;
    }

    setGcLoading(true);

    try {
      const payload = {
        dicom_path: gradcamDicom,
        class_idx: Number(gradcamClass),
      };

      const res = await fetch(`${API_BASE}/gradcam/ct_slice`, {
        method: "POST",
        headers: authHeaders({ "Content-Type": "application/json" }),
        body: JSON.stringify(payload),
      });

      handleUnauthorized(res.status);

      if (!res.ok) {
        const detail = await res.json().catch(() => ({}));
        throw new Error(detail.detail || `HTTP ${res.status}`);
      }

      const data = await res.json();
      setGradcamUrl(`${API_BASE}${data.gradcam_url}`);
    } catch (e2) {
      setGcErr(e2?.message || "Grad-CAM failed");
    } finally {
      setGcLoading(false);
    }
  }

  async function downloadPdf() {
    setPdfErr("");

    if (!result) {
      setPdfErr("Run triage first, then export the report.");
      return;
    }

    setPdfLoading(true);

    try {
      const payload = {
        ct_dir: ctDir || null,
        mri_dwi: mriDwi || null,
        mri_adc: mriAdc || null,
        mri_flair: mriFlair || null,

        patient_id: patientId || null,
        study_id: studyId || null,
        generated_by: "Stroke AI Triage System",

        include_gradcam: Boolean(gradcamDicom),
        gradcam_dicom_path: gradcamDicom || null,
        gradcam_class_idx: Number(gradcamClass),
      };

      const res = await fetch(`${API_BASE}/report/pdf`, {
        method: "POST",
        headers: authHeaders({ "Content-Type": "application/json" }),
        body: JSON.stringify(payload),
      });

      handleUnauthorized(res.status);

      if (!res.ok) {
        const detail = await res.json().catch(() => ({}));
        throw new Error(detail.detail || `HTTP ${res.status}`);
      }

      const blob = await res.blob();
      const url = window.URL.createObjectURL(blob);

      const ts = new Date().toISOString().replace(/[:.]/g, "-");
      const nameParts = [
        "stroke_triage_report",
        safeFileName(patientId || "patient"),
        safeFileName(studyId || "study"),
        ts,
      ];
      const filename = `${nameParts.join("_")}.pdf`;

      const a = document.createElement("a");
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      a.remove();
      window.URL.revokeObjectURL(url);
    } catch (e2) {
      setPdfErr(e2?.message || "PDF export failed");
    } finally {
      setPdfLoading(false);
    }
  }

  if (!token) {
    return (
      <Login
        apiBase={API_BASE}
        onLogin={(newToken, username) => {
          localStorage.setItem("stroke_ai_token", newToken);
          localStorage.setItem("stroke_ai_admin", username);
          setToken(newToken);
          setAdminUser(username);
        }}
      />
    );
  }

  const hemSubtypes = result?.hemorrhage?.subtypes || {};
  const reasons = result?.reasons || [];
  const conf = result?.confidence_summary || {};

  const ischemiaVolMl =
    result?.ischemia?.lesion_volume_ml != null
      ? Number(result.ischemia.lesion_volume_ml).toFixed(2)
      : "0.00";

  const hemAnyStr =
    conf?.hem_any != null ? Number(conf.hem_any).toFixed(2) : "0.00";

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      <div className="sticky top-0 z-20 border-b border-slate-800 bg-slate-950/80 backdrop-blur">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-4 py-3">
          <div className="flex items-center gap-3">
            <div className="text-xl font-extrabold">🧠 Stroke AI Triage</div>
            <Badge tone={apiUp ? "green" : "red"}>
              <span
                className={cls(
                  "h-2 w-2 rounded-full",
                  apiUp ? "bg-emerald-400" : "bg-red-400"
                )}
              />
              {apiUp ? "API Online" : "API Offline"}
            </Badge>
            <Badge tone="blue">
              Admin: {adminUser || "admin"}
            </Badge>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <a
              className="rounded-lg border border-slate-700 px-3 py-1.5 text-xs font-semibold text-slate-200 hover:bg-slate-900"
              href={`${API_BASE}/docs`}
              target="_blank"
              rel="noreferrer"
            >
              Open API Docs
            </a>

            {result && (
              <button
                className="rounded-lg border border-slate-700 px-3 py-1.5 text-xs font-semibold text-slate-200 hover:bg-slate-900"
                onClick={() => copyToClipboard(JSON.stringify(result, null, 2))}
              >
                Copy JSON
              </button>
            )}

            <button
              className={cls(
                "rounded-lg border px-3 py-1.5 text-xs font-semibold",
                result
                  ? "border-slate-700 text-slate-200 hover:bg-slate-900"
                  : "cursor-not-allowed border-slate-800 text-slate-500",
                pdfLoading ? "opacity-70" : ""
              )}
              onClick={() => downloadPdf()}
              disabled={!result || pdfLoading}
              title={!result ? "Run triage first" : "Download PDF report"}
            >
              {pdfLoading ? "Generating PDF..." : "Download PDF"}
            </button>

            <button
              className="rounded-lg border border-red-500/30 px-3 py-1.5 text-xs font-semibold text-red-200 hover:bg-red-500/10"
              onClick={handleLogout}
            >
              Logout
            </button>
          </div>
        </div>
      </div>

      <div className="mx-auto grid max-w-6xl grid-cols-1 gap-4 px-4 py-6 lg:grid-cols-12">
        <div className="lg:col-span-5">
          <div className="rounded-2xl border border-slate-800 bg-slate-900/30 p-4">
            <div className="mb-3 flex items-center justify-between">
              <div className="text-sm font-bold text-slate-200">Inputs</div>
              <Badge tone="purple">Local paths mode</Badge>
            </div>

            <form onSubmit={runTriage} className="space-y-3">
              <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
                <div>
                  <label className="text-xs font-semibold text-slate-300">
                    Patient ID (optional)
                  </label>
                  <input
                    className="mt-1 w-full rounded-xl border border-slate-700 bg-slate-950/60 px-3 py-2 text-sm outline-none focus:border-sky-500"
                    value={patientId}
                    onChange={(e) => setPatientId(e.target.value)}
                    placeholder="P001"
                  />
                </div>

                <div>
                  <label className="text-xs font-semibold text-slate-300">
                    Study ID (optional)
                  </label>
                  <input
                    className="mt-1 w-full rounded-xl border border-slate-700 bg-slate-950/60 px-3 py-2 text-sm outline-none focus:border-sky-500"
                    value={studyId}
                    onChange={(e) => setStudyId(e.target.value)}
                    placeholder="Study001"
                  />
                </div>
              </div>

              <div>
                <label className="text-xs font-semibold text-slate-300">
                  CT DICOM directory (one exam)
                </label>
                <input
                  className="mt-1 w-full rounded-xl border border-slate-700 bg-slate-950/60 px-3 py-2 text-sm outline-none focus:border-sky-500"
                  value={ctDir}
                  onChange={(e) => setCtDir(e.target.value)}
                  placeholder="K:\stroke_ai\data\rsna\sample_exam"
                />
                <div className="mt-1 text-[11px] text-slate-400">
                  Tip: don’t pass the whole stage_2_test folder; use a single exam folder.
                </div>
              </div>

              <div className="grid grid-cols-1 gap-3 md:grid-cols-3">
                <div>
                  <label className="text-xs font-semibold text-slate-300">
                    MRI DWI
                  </label>
                  <input
                    className="mt-1 w-full rounded-xl border border-slate-700 bg-slate-950/60 px-3 py-2 text-sm outline-none focus:border-sky-500"
                    value={mriDwi}
                    onChange={(e) => setMriDwi(e.target.value)}
                    placeholder="..._dwi.nii.gz"
                  />
                </div>

                <div>
                  <label className="text-xs font-semibold text-slate-300">
                    MRI ADC
                  </label>
                  <input
                    className="mt-1 w-full rounded-xl border border-slate-700 bg-slate-950/60 px-3 py-2 text-sm outline-none focus:border-sky-500"
                    value={mriAdc}
                    onChange={(e) => setMriAdc(e.target.value)}
                    placeholder="..._adc.nii.gz"
                  />
                </div>

                <div>
                  <label className="text-xs font-semibold text-slate-300">
                    MRI FLAIR
                  </label>
                  <input
                    className="mt-1 w-full rounded-xl border border-slate-700 bg-slate-950/60 px-3 py-2 text-sm outline-none focus:border-sky-500"
                    value={mriFlair}
                    onChange={(e) => setMriFlair(e.target.value)}
                    placeholder="..._FLAIR.nii.gz"
                  />
                </div>
              </div>

              {err && (
                <div className="rounded-xl border border-red-500/30 bg-red-500/10 px-3 py-2 text-sm text-red-200">
                  {err}
                </div>
              )}

              {pdfErr && (
                <div className="rounded-xl border border-orange-500/30 bg-orange-500/10 px-3 py-2 text-sm text-orange-200">
                  {pdfErr}
                </div>
              )}

              <button
                type="submit"
                disabled={loading}
                className={cls(
                  "w-full rounded-xl px-4 py-2 text-sm font-extrabold",
                  loading
                    ? "bg-slate-700 text-slate-200"
                    : "bg-emerald-500 text-slate-950 hover:bg-emerald-400"
                )}
              >
                {loading ? "Running triage..." : "Run AI Triage"}
              </button>
            </form>
          </div>

          <div className="mt-4 rounded-2xl border border-slate-800 bg-slate-900/30 p-4">
            <div className="mb-3 flex items-center justify-between">
              <div className="text-sm font-bold text-slate-200">
                Explainability (CT Grad-CAM)
              </div>
              <Badge tone="blue">Slice-level</Badge>
            </div>

            <form onSubmit={runGradcam} className="space-y-3">
              <div>
                <label className="text-xs font-semibold text-slate-300">
                  CT DICOM slice path
                </label>
                <input
                  className="mt-1 w-full rounded-xl border border-slate-700 bg-slate-950/60 px-3 py-2 text-sm outline-none focus:border-sky-500"
                  value={gradcamDicom}
                  onChange={(e) => setGradcamDicom(e.target.value)}
                  placeholder="K:\stroke_ai\data\rsna\stage_2_train\ID_xxx.dcm"
                />
              </div>

              <div className="flex items-center gap-2">
                <select
                  className="rounded-xl border border-slate-700 bg-slate-950/60 px-3 py-2 text-sm outline-none focus:border-sky-500"
                  value={gradcamClass}
                  onChange={(e) => setGradcamClass(Number(e.target.value))}
                >
                  {RSNA_CLASSES.map((c) => (
                    <option key={c.value} value={c.value}>
                      {c.value} — {c.label}
                    </option>
                  ))}
                </select>

                <button
                  type="submit"
                  disabled={gcLoading}
                  className={cls(
                    "rounded-xl px-4 py-2 text-sm font-extrabold",
                    gcLoading
                      ? "bg-slate-700 text-slate-200"
                      : "bg-sky-500 text-slate-950 hover:bg-sky-400"
                  )}
                >
                  {gcLoading ? "Generating..." : "Generate"}
                </button>
              </div>

              {gcErr && (
                <div className="rounded-xl border border-red-500/30 bg-red-500/10 px-3 py-2 text-sm text-red-200">
                  {gcErr}
                </div>
              )}

              {gradcamUrl && (
                <div className="overflow-hidden rounded-2xl border border-slate-800 bg-slate-950/40">
                  <img src={gradcamUrl} alt="Grad-CAM" className="w-full object-contain" />
                </div>
              )}

              <div className="text-[11px] text-slate-400">
                Note: If you export PDF after generating, the report can embed Grad-CAM.
              </div>
            </form>
          </div>
        </div>

        <div className="lg:col-span-7">
          <div className="rounded-2xl border border-slate-800 bg-slate-900/30 p-4">
            <div className="mb-3 flex items-center justify-between">
              <div className="text-sm font-bold text-slate-200">Result</div>
              {result ? (
                <div className="flex flex-wrap items-center gap-2">
                  <Badge tone={severityTone(result.severity)}>
                    {String(result.severity).toUpperCase()}
                  </Badge>

                  {result.review_priority && (
                    <Badge tone={reviewTone(result.review_priority)}>
                      {result.review_priority.replaceAll("_", " ").toUpperCase()}
                    </Badge>
                  )}

                  <Badge tone={result.stroke_present ? "green" : "gray"}>
                    Stroke: {result.stroke_present ? "YES" : "NO"}
                  </Badge>

                  <Badge tone="purple">{result.stroke_type}</Badge>
                </div>
              ) : (
                <Badge tone="gray">No result yet</Badge>
              )}
            </div>

            {!result ? (
              <div className="rounded-xl border border-slate-800 bg-slate-950/40 p-6 text-sm text-slate-300">
                Run triage to see predictions, reasons, and confidence.
              </div>
            ) : (
              <div className="grid grid-cols-1 gap-4">
                <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
                  <div className="rounded-2xl border border-slate-800 bg-slate-950/40 p-3">
                    <div className="text-[11px] text-slate-400">CT Slices</div>
                    <div className="text-lg font-extrabold">
                      {result?.hemorrhage?.meta?.num_slices ?? 0}
                    </div>
                  </div>

                  <div className="rounded-2xl border border-slate-800 bg-slate-950/40 p-3">
                    <div className="text-[11px] text-slate-400">Hem Any</div>
                    <div className="text-lg font-extrabold">{hemAnyStr}</div>
                  </div>

                  <div className="rounded-2xl border border-slate-800 bg-slate-950/40 p-3">
                    <div className="text-[11px] text-slate-400">Ischemia Vol (ml)</div>
                    <div className="text-lg font-extrabold">{ischemiaVolMl}</div>
                  </div>

                  <div className="rounded-2xl border border-slate-800 bg-slate-950/40 p-3">
                    <div className="text-[11px] text-slate-400">Borderline</div>
                    <div className="text-lg font-extrabold">
                      {conf.borderline ? "YES" : "NO"}
                    </div>
                  </div>
                </div>

                <div className="rounded-2xl border border-slate-800 bg-slate-950/40 p-3">
                  <div className="mb-2 text-xs font-bold text-slate-300">Reasons</div>
                  {reasons.length === 0 ? (
                    <div className="text-sm text-slate-400">No reasons returned.</div>
                  ) : (
                    <ul className="space-y-1 text-sm text-slate-200">
                      {reasons.map((r, i) => (
                        <li key={i} className="flex gap-2">
                          <span className="mt-1 h-2 w-2 rounded-full bg-sky-400" />
                          <span>{r}</span>
                        </li>
                      ))}
                    </ul>
                  )}
                </div>

                <div className="rounded-2xl border border-slate-800 bg-slate-950/40 p-3">
                  <div className="mb-2 text-xs font-bold text-slate-300">
                    Hemorrhage probabilities
                  </div>

                  <div className="space-y-2">
                    {Object.entries(hemSubtypes).map(([k, v]) => (
                      <div key={k} className="grid grid-cols-12 items-center gap-2">
                        <div className="col-span-3 text-xs font-semibold capitalize text-slate-300">
                          {k}
                        </div>
                        <div className="col-span-7">
                          <ProgressBar value={Number(v) || 0} />
                        </div>
                        <div className="col-span-2 text-right text-xs font-semibold text-slate-200">
                          {Math.round((Number(v) || 0) * 100)}%
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="rounded-2xl border border-slate-800 bg-slate-950/40 p-3">
                  <div className="mb-2 flex items-center justify-between">
                    <div className="text-xs font-bold text-slate-300">
                      Ischemia output
                    </div>

                    {result?.ischemia?.mask_path && (
                      <button
                        className="rounded-lg border border-slate-700 px-2 py-1 text-xs font-semibold text-slate-200 hover:bg-slate-900"
                        onClick={() => copyToClipboard(result.ischemia.mask_path)}
                      >
                        Copy mask path
                      </button>
                    )}
                  </div>

                  <div className="text-sm text-slate-200">
                    Lesion volume: <span className="font-bold">{ischemiaVolMl} ml</span>
                  </div>

                  <div className="mt-1 break-all text-xs text-slate-400">
                    Mask path: {result?.ischemia?.mask_path || "N/A"}
                  </div>
                </div>

                <details className="rounded-2xl border border-slate-800 bg-slate-950/40 p-3">
                  <summary className="cursor-pointer text-xs font-bold text-slate-300">
                    Raw JSON
                  </summary>
                  <pre className="mt-2 max-h-72 overflow-auto rounded-xl bg-slate-950/60 p-3 text-[11px] text-slate-300">
                    {JSON.stringify(result, null, 2)}
                  </pre>
                </details>
              </div>
            )}
          </div>
        </div>
      </div>

      <div className="border-t border-slate-900 py-6 text-center text-xs text-slate-500">
        Research prototype — validate clinically before real-world use.
      </div>
    </div>
  );
}