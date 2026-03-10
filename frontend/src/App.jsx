import { useState, useEffect, useRef } from "react";
import ReactMarkdown from "react-markdown";

const API_BASE = "http://localhost:8000";

/* ── Global styles ── */
const G = `
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Syne:wght@700;800&display=swap');
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: #0a0e14;
    font-family: 'Inter', sans-serif;
    color: #cbd5e1;
    -webkit-font-smoothing: antialiased;
  }
  ::-webkit-scrollbar { width: 4px; }
  ::-webkit-scrollbar-thumb { background: #1e293b; border-radius: 4px; }
  input, textarea, button { font-family: inherit; }

  @keyframes spin    { to { transform: rotate(360deg) } }
  @keyframes pulse   { 0%,100%{opacity:1} 50%{opacity:.2} }
  @keyframes fadeUp  { from{opacity:0;transform:translateY(12px)} to{opacity:1;transform:translateY(0)} }
  @keyframes slideIn { from{opacity:0;transform:translateX(14px)} to{opacity:1;transform:translateX(0)} }
  @keyframes chatIn  { from{opacity:0;transform:translateY(5px)}  to{opacity:1;transform:translateY(0)} }
  @keyframes scan    { 0%{top:-4px} 100%{top:100%} }

  .fu  { animation: fadeUp  .38s ease both; }
  .fu2 { animation: fadeUp  .38s .08s ease both; }
  .si  { animation: slideIn .36s ease both; }

  /* ── Report typography ── */
  .report-body { padding: 24px 28px; }
  .report-body h2 {
    font-family: 'Syne', sans-serif;
    font-size: .98rem;
    font-weight: 700;
    color: #f8fafc;
    margin: 24px 0 9px;
    padding-left: 12px;
    border-left: 2px solid #3b82f6;
  }
  .report-body h2:first-child { margin-top: 0; }
  .report-body p {
    font-size: .83rem;
    color: #94a3b8;
    line-height: 1.82;
    margin-bottom: 5px;
  }
  .report-body ul {
    margin: 4px 0 12px 16px;
    display: flex;
    flex-direction: column;
    gap: 5px;
    list-style: none;
  }
  .report-body li {
    font-size: .83rem;
    color: #94a3b8;
    line-height: 1.72;
    position: relative;
    padding-left: 16px;
  }
  .report-body li::before {
    content: '•';
    position: absolute;
    left: 0;
    color: #3b82f6;
  }
  .report-body strong { 
    color: rgba(253, 230, 138, 0.9); 
    background: rgba(120, 53, 15, 0.3);
    padding: 2px 6px;
    border-radius: 4px;
    font-weight: 500; 
  }
`;

/* ── Design tokens — matching high-contrast dark theme ── */
const C = {
  page:     "#0a0e14",
  panel:    "#111823",
  header:   "#161f2c",
  card:     "#111823",
  cardB:    "#1a2433",
  border:   "#1e293b",
  borderB:  "#334155",
  accent:   "#2563eb",
  accentB:  "#60a5fa",
  accentDim:"rgba(59, 130, 246, 0.2)",
  txt:      "#f8fafc",
  txts:     "#94a3b8",
  txtt:     "#64748b",
  ok:       "#34d399",
  okDim:    "rgba(52, 211, 153, 0.1)",
  okBdr:    "rgba(52, 211, 153, 0.2)",
  warn:     "#fbbf24",
  warnDim:  "rgba(251, 191, 36, 0.1)",
  warnBdr:  "rgba(251, 191, 36, 0.2)",
  err:      "#f87171",
  errDim:   "rgba(248, 113, 113, 0.1)",
  errBdr:   "rgba(248, 113, 113, 0.2)",
};

/* ── Primitives ── */
const Spinner = ({ size = 16, color = C.accentB }) => (
  <span style={{ display:"inline-block", width:size, height:size, border:`2px solid ${color}22`, borderTopColor:color, borderRadius:"50%", animation:"spin .7s linear infinite", flexShrink:0 }} />
);

function Btn({ children, onClick, disabled, loading, ghost, full, sm }) {
  const [h, setH] = useState(false);
  return (
    <button onClick={onClick} disabled={disabled||loading}
      onMouseEnter={()=>setH(true)} onMouseLeave={()=>setH(false)}
      style={{
        width: full?"100%":"auto",
        display:"inline-flex", alignItems:"center", justifyContent:"center", gap:7,
        padding: sm ? ".38rem .9rem" : ".62rem 1.45rem",
        borderRadius:8, fontWeight:600,
        fontSize: sm ? ".75rem" : ".85rem",
        border: ghost ? `1px solid ${h ? C.accentB : C.border}` : "none",
        cursor: (disabled||loading) ? "not-allowed" : "pointer",
        background: ghost
          ? (h ? "rgba(96,165,250,.1)" : "transparent")
          : (disabled||loading) ? C.cardB : h ? C.accentB : C.accent,
        color: ghost
          ? (h ? C.accentB : C.txts)
          : (disabled||loading) ? C.txtt : "#fff",
        boxShadow: !ghost && !(disabled||loading)
          ? (h ? "0 2px 18px rgba(37,99,235,.45)" : "0 2px 10px rgba(37,99,235,.28)")
          : "none",
        transform: !ghost && !(disabled||loading) && h ? "translateY(-1px)" : "none",
        transition: "all .15s ease",
      }}>
      {loading && <Spinner size={13} color={ghost ? C.accentB : "#fff"} />}
      {children}
    </button>
  );
}

const Panel = ({ children, style={}, cls="" }) => (
  <div className={cls} style={{
    background: C.panel,
    border: `1px solid ${C.border}`,
    borderRadius: 12,
    boxShadow: "0 10px 30px rgba(0,0,0,.5)",
    overflow: "hidden",
    ...style,
  }}>{children}</div>
);

const SectionLabel = ({ children }) => (
  <p style={{ fontSize:".67rem", fontWeight:700, letterSpacing:"1px", textTransform:"uppercase", color:C.txtt, marginBottom:10 }}>{children}</p>
);

function parseReport(text) {
  if (!text) return [];
  const lines  = text.split("\n");
  const blocks = [];
  let listItems = [];

  const flush = () => {
    if (listItems.length) { blocks.push({ type:"list", items:[...listItems] }); listItems = []; }
  };

  const isHeader = (line) => {
    const t = line.trim();
    if (!t || t.length < 3) return false;
    if (t === t.toUpperCase() && /^[A-Z\s\/\-:]+$/.test(t)) return true;
    if (/^[A-Z][a-zA-Z\s]+:$/.test(t)) return true;
    if (/^(Findings|Impression|Clinical History|Technique|Conclusion|Summary|Recommendations?|History|Indication|Protocol|Results?|Assessment|Plan|Discussion)\s*:?$/i.test(t)) return true;
    return false;
  };

  for (const raw of lines) {
    const line = raw.trimEnd();
    if (!line.trim()) { flush(); continue; }
    if (isHeader(line)) {
      flush();
      blocks.push({ type:"heading", text: line.trim().replace(/:$/, "") });
    } else if (/^\s*[-•*]\s+/.test(line)) {
      listItems.push(line.replace(/^\s*[-•*]\s+/, "").trim());
    } else if (/^\s*\d+\.\s+/.test(line)) {
      listItems.push(line.replace(/^\s*\d+\.\s+/, "").trim());
    } else {
      flush();
      blocks.push({ type:"para", text: line.trim() });
    }
  }
  flush();
  return blocks;
}

function renderInline(text) {
  return text.split(/(\*\*[^*]+\*\*)/).map((p, i) =>
    p.startsWith("**") ? <strong key={i}>{p.slice(2,-2)}</strong> : p
  );
}

function ReportRenderer({ text }) {
  const blocks = parseReport(text);
  return (
    <div className="report-body">
      {blocks.map((b, i) => {
        if (b.type === "heading") return <h2 key={i}>{b.text}</h2>;
        if (b.type === "list")    return <ul key={i}>{b.items.map((item,j) => <li key={j}>{renderInline(item)}</li>)}</ul>;
        return <p key={i}>{renderInline(b.text)}</p>;
      })}
    </div>
  );
}

const REQUIRED = ["t1","t1ce","t2","flair"];
const ALIASES  = { t1n:"t1", t1c:"t1ce", t2w:"t2", t2f:"flair" };
function detectMods(files) {
  const found = new Set();
  for (const f of files) {
    const stem   = f.name.toLowerCase().replace(/\.nii\.gz$|\.nii$/, "");
    const parts  = stem.split(/[_-]/);
    const suffix = parts[parts.length-1];
    found.add(ALIASES[suffix] ?? suffix);
  }
  return found;
}

export default function App() {
  const [files,     setFiles]     = useState([]);
  const [caseId,    setCaseId]    = useState("");
  const [dragging,  setDragging]  = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadErr, setUploadErr] = useState("");
  const [jobId,     setJobId]     = useState(null);
  const [progress,  setProgress]  = useState(0);
  const [stage,     setStage]     = useState("");
  const [report,    setReport]    = useState("");
  const [pdfUrl,    setPdfUrl]    = useState("");
  const [reportTab, setReportTab] = useState("pdf");
  const [question,  setQuestion]  = useState("");
  const [asking,    setAsking]    = useState(false);
  const [history,   setHistory]   = useState([]);
  const fileRef    = useRef();
  const chatEndRef = useRef();

  const phase = !jobId ? "upload" : report ? "done" : "processing";

  useEffect(() => {
    if (phase !== "processing") return;
    let timer;
    const poll = async () => {
      try {
        const r = await fetch(`${API_BASE}/api/status/${jobId}`);
        const d = await r.json();
        setProgress(Math.min(d.progress_pct ?? 0, 100));
        setStage(d.stage ?? "");
        if (d.status === "done") {
          const rr = await fetch(`${API_BASE}/api/report/${jobId}`);
          setReport(await rr.text());
          try {
            const pr = await fetch(`${API_BASE}/api/report/${jobId}/pdf`);
            if (pr.ok) { const blob = await pr.blob(); setPdfUrl(URL.createObjectURL(blob)); }
          } catch {}
          return;
        }
        if (d.status === "error") { setUploadErr(d.error_message || "Pipeline error."); setJobId(null); return; }
      } catch {}
      timer = setTimeout(poll, 4000);
    };
    poll();
    return () => clearTimeout(timer);
  }, [phase, jobId]);

  useEffect(() => { chatEndRef.current?.scrollIntoView({ behavior:"smooth" }); }, [history, asking]);

  const handleFiles = (list) => { setFiles(Array.from(list)); setUploadErr(""); };

  const submit = async () => {
    setUploading(true); setUploadErr("");
    try {
      const fd = new FormData();
      fd.append("case_id", caseId.trim());
      for (const f of files) fd.append("files", f);
      const r = await fetch(`${API_BASE}/api/analyze`, { method:"POST", body:fd });
      if (!r.ok) throw new Error((await r.json()).detail ?? "Upload failed");
      const { job_id } = await r.json();
      setJobId(job_id); setProgress(0);
    } catch(e) { setUploadErr(e.message); }
    finally { setUploading(false); }
  };

  const ask = async () => {
    if (!question.trim() || asking) return;
    const q = question.trim(); setQuestion(""); setAsking(true);
    try {
      const r = await fetch(`${API_BASE}/api/chat/${jobId}`, {
        method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({ question:q }),
      });
      const d = await r.json();
      setHistory(h => [...h, { q, a: d.answer || "No answer returned." }]);
    } catch(e) { setHistory(h => [...h, { q, a:"Error: "+e.message }]); }
    finally { setAsking(false); }
  };

  const reset = () => {
    if (pdfUrl) URL.revokeObjectURL(pdfUrl);
    setFiles([]); setCaseId(""); setJobId(null); setReport(""); setPdfUrl("");
    setProgress(0); setStage(""); setHistory([]); setQuestion(""); setUploadErr(""); setReportTab("pdf");
  };

  const found      = detectMods(files);
  const missing    = REQUIRED.filter(m => !found.has(m));
  const allPresent = files.length > 0 && missing.length === 0;
  const ready      = allPresent && caseId.trim() && !uploading;

  const panelHeader = {
    padding:"16px 24px",
    background: C.header,
    borderBottom:`1px solid ${C.border}`,
    display:"flex", alignItems:"center", justifyContent:"space-between",
    flexShrink:0,
  };

  return (
    <>
      <style>{G}</style>

      <div style={{ minHeight:"100vh", display:"flex", flexDirection:"column", background:C.page }}>

        <header style={{
          height:64, flexShrink:0,
          background: C.page,
          padding: "0 32px",
          display:"flex", alignItems:"center", justifyContent:"space-between",
          position:"sticky", top:0, zIndex:100,
        }}>
          <div style={{ display:"flex", alignItems:"center", gap:10 }}>
            <span style={{ fontSize:"1.4rem" }}>🧠</span>
            <span style={{ fontFamily:"'Syne',sans-serif", fontWeight:800, fontSize:"1.1rem", color:"#f8fafc", letterSpacing:"-.1px" }}>
              AI Powered Brain MRI Assistant 
            </span>
          </div>

          <div style={{ display:"flex", alignItems:"center", gap:10 }}>
            {phase === "processing" && (
              <div style={{ display:"flex", alignItems:"center", gap:7, background:C.warnDim, border:`1px solid ${C.warnBdr}`, borderRadius:50, padding:"4px 13px" }}>
                <span style={{ width:6, height:6, borderRadius:"50%", background:C.warn, display:"inline-block", animation:"pulse 1.2s ease infinite" }} />
                <span style={{ color:C.warn, fontSize:".72rem", fontWeight:600 }}>Analysis running</span>
              </div>
            )}
            {phase === "done" && (
              <div style={{ display:"flex", alignItems:"center", gap:7, background:C.okDim, border:`1px solid ${C.okBdr}`, borderRadius:50, padding:"4px 13px" }}>
                <span style={{ color:C.ok, fontSize:".72rem", fontWeight:600 }}>✓ Report ready</span>
              </div>
            )}
            {phase === "done" && <Btn ghost sm onClick={reset}>↺ New analysis</Btn>}
          </div>
        </header>

        <main style={{ flex:1, padding:"10px 32px 44px", display:"flex", flexDirection:"column", gap:18, maxWidth: 1600, margin: "0 auto", width: "100%" }}>

          <section style={{
            opacity: phase === "done" ? 0 : 1,
            display: phase === "done" ? "none" : "block",
            pointerEvents: phase === "done" ? "none" : "auto",
            transition: "opacity .5s",
          }}>
            <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:24 }}>
              <Panel cls="fu" style={{ borderStyle: "dashed", background: "transparent", borderColor: C.borderB }}>
                <div style={{ padding:"40px 32px", height: "100%", display: "flex", flexDirection: "column", justifyContent: "center" }}>
                  <SectionLabel>MRI Sequences</SectionLabel>
                  <div
                    onClick={() => fileRef.current?.click()}
                    onDragOver={e => { e.preventDefault(); setDragging(true); }}
                    onDragLeave={() => setDragging(false)}
                    onDrop={e => { e.preventDefault(); setDragging(false); handleFiles(e.dataTransfer.files); }}
                    style={{
                      background: dragging ? C.accentDim : "rgba(255,255,255,.02)",
                      border: `2px dashed ${dragging ? C.accentB : C.borderB}`,
                      borderRadius:12, padding: files.length ? "20px" : "60px 20px",
                      textAlign:"center", cursor:"pointer", transition:"all .2s",
                      position:"relative", overflow:"hidden",
                      flex: 1, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center"
                    }}
                  >
                    {dragging && (
                      <div style={{ position:"absolute", left:0, right:0, height:2, background:`linear-gradient(90deg,transparent,${C.accentB},transparent)`, animation:"scan 1s linear infinite" }} />
                    )}
                    {files.length === 0 ? (
                      <>
                        <div style={{ marginBottom: 16, color: C.txtt }}>
                          <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" x2="12" y1="3" y2="15"/></svg>
                        </div>
                        <p style={{ color:C.txt, fontWeight:500, fontSize:".95rem" }}>
                          Drag and drop MRI scans
                        </p>
                        <p style={{ color:C.txtt, fontSize:".8rem", marginTop:8 }}>Files are encrypted and processed locally (DICOM / .nii.gz)</p>
                      </>
                    ) : (
                      <div style={{ textAlign:"left", width: "100%" }}>
                        <p style={{ fontSize:".85rem", fontWeight:600, color:C.txts, marginBottom:12 }}>
                          {files.length} file{files.length > 1 ? "s" : ""} selected —{" "}
                          <span onClick={e => { e.stopPropagation(); setFiles([]); }} style={{ color:C.accentB, cursor:"pointer" }}>clear</span>
                        </p>
                        <div style={{ display:"flex", flexWrap:"wrap", gap:8 }}>
                          {files.map(f => (
                            <span key={f.name} style={{ background:C.accentDim, color:C.accentB, border:`1px solid rgba(59, 130, 246, 0.3)`, borderRadius:6, padding:"4px 10px", fontSize:".75rem", fontWeight:600 }}>{f.name}</span>
                          ))}
                        </div>
                      </div>
                    )}
                    <input ref={fileRef} type="file" accept=".nii.gz,.nii" multiple style={{ display:"none" }} onChange={e => handleFiles(e.target.files)} />
                  </div>

                  {files.length > 0 && (
                    <div style={{ display:"flex", gap:8, marginTop:16 }}>
                      {REQUIRED.map(mod => {
                        const ok = found.has(mod);
                        return (
                          <div key={mod} style={{ flex:1, textAlign:"center", padding:"8px 4px", borderRadius:8, fontSize:".75rem", fontWeight:700, background: ok ? C.okDim : C.warnDim, color: ok ? C.ok : C.warn, border:`1px solid ${ok ? C.okBdr : C.warnBdr}`, transition:"all .25s" }}>
                            {ok ? "✓" : "✗"} {mod.toUpperCase()}
                          </div>
                        );
                      })}
                    </div>
                  )}
                </div>
              </Panel>

              <Panel cls="fu2" style={{ display:"flex", flexDirection:"column" }}>
                <div style={{ padding:"40px 32px", flex:1, display:"flex", flexDirection:"column", gap:20 }}>
                  <div>
                    <SectionLabel>Case Details</SectionLabel>
                    <label style={{ display:"block", fontSize:".85rem", fontWeight:500, color:C.txts, marginBottom:8 }}>Patient / Case ID</label>
                    <input
                      value={caseId} onChange={e => setCaseId(e.target.value)}
                      placeholder="e.g. BraTS-GLI-00003-000"
                      style={{ width:"100%", padding:"12px 16px", borderRadius:8, border:`1px solid ${C.border}`, background:C.cardB, fontSize:".9rem", color:C.txt, outline:"none", transition:"border-color .18s, box-shadow .18s" }}
                      onFocus={e => { e.target.style.borderColor = C.accentB; e.target.style.boxShadow = "0 0 0 3px rgba(59,130,246,.16)"; }}
                      onBlur={e  => { e.target.style.borderColor = C.border;  e.target.style.boxShadow = "none"; }}
                    />
                  </div>

                  <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:12 }}>
                    {[
                      { label:"Segmentation", val:"nnUNet v2" },
                      { label:"Processing",   val:"3 – 5 min" },
                    ].map(({ label, val }) => (
                      <div key={label} style={{ background:C.cardB, border:`1px solid ${C.border}`, borderRadius:8, padding:"14px 16px" }}>
                        <p style={{ fontSize:".7rem", fontWeight:600, letterSpacing:".5px", textTransform:"uppercase", color:C.txtt, marginBottom:4 }}>{label}</p>
                        <p style={{ fontSize:".9rem", fontWeight:700, color:C.txt }}>{val}</p>
                      </div>
                    ))}
                  </div>

                  {uploadErr && (
                    <div style={{ background:C.errDim, border:`1px solid ${C.errBdr}`, borderRadius:8, padding:"12px 16px", fontSize:".85rem", color:C.err }}>⚠ {uploadErr}</div>
                  )}

                  <div style={{ marginTop:"auto" }}>
                    <Btn full onClick={submit} disabled={!ready} loading={uploading}>
                      {uploading ? "Uploading…" : "Run Analysis →"}
                    </Btn>
                  </div>
                </div>
              </Panel>
            </div>
          </section>

          {phase === "processing" && (
            <Panel cls="fu" style={{ padding:"24px 32px" }}>
              <div style={{ display:"flex", alignItems:"center", gap:20 }}>
                <Spinner size={28} color={C.accentB} />
                <div style={{ flex:1 }}>
                  <div style={{ display:"flex", justifyContent:"space-between", marginBottom:10 }}>
                    <p style={{ fontWeight:600, fontSize:"1rem", color:C.txt }}>
                      {stage ? stage.charAt(0).toUpperCase() + stage.slice(1) + "…" : "Processing…"}
                    </p>
                    <p style={{ fontWeight:700, fontSize:"1rem", color:C.accentB }}>{progress}%</p>
                  </div>
                  <div style={{ height:6, background:C.border, borderRadius:50, overflow:"hidden" }}>
                    <div style={{ height:"100%", width:`${progress}%`, background:`linear-gradient(90deg,${C.accent},${C.accentB})`, borderRadius:50, transition:"width .6s ease" }} />
                  </div>
                  <p style={{ fontSize:".8rem", color:C.txtt, marginTop:8 }}>Analysis typically completes in 3 – 5 minutes. Keep this tab open.</p>
                </div>
              </div>
            </Panel>
          )}

          {phase === "done" && (
            <div style={{ display:"grid", gridTemplateColumns:"1.5fr 1fr", gap:24, animation:"fadeUp .4s ease both", flex:1, minHeight: 0 }}>

              <Panel style={{ display:"flex", flexDirection:"column" }}>
                <div style={panelHeader}>
                  <div style={{ display:"flex", alignItems:"center", gap:10 }}>
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke={C.accentB} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z"/><polyline points="14 2 14 8 20 8"/><line x1="16" x2="8" y1="13" y2="13"/><line x1="16" x2="8" y1="17" y2="17"/><line x1="10" x2="8" y1="9" y2="9"/></svg>
                    <span style={{ fontFamily:"'Inter',sans-serif", fontWeight:600, fontSize:".95rem", color:C.txts, textTransform: "uppercase", letterSpacing: "1px" }}>Radiological Report</span>
                  </div>
                  <div style={{ display:"flex", gap:8 }}>
                    {pdfUrl && (
                      <Btn ghost sm onClick={() => { const a=document.createElement("a"); a.href=pdfUrl; a.download=`${caseId}.pdf`; a.click(); }}>⬇ PDF</Btn>
                    )}
                    <Btn ghost sm onClick={() => {
                      const b = new Blob([report], { type:"text/plain" });
                      const a = document.createElement("a"); a.href = URL.createObjectURL(b); a.download = `${caseId}.txt`; a.click();
                    }}>⬇ Text</Btn>
                  </div>
                </div>

                <div style={{ display:"flex", padding:"16px 24px 0", gap:0, flexShrink:0, borderBottom: `1px solid ${C.border}` }}>
                  {[
                    { key:"pdf",  label:"📄 PDF Report" },
                    { key:"text", label:"📝 Text Report" },
                  ].map(t => (
                    <button key={t.key} onClick={() => setReportTab(t.key)} style={{
                      padding:"8px 20px", fontSize:".85rem", fontWeight:600, cursor:"pointer",
                      background: reportTab === t.key ? C.accentDim : "transparent",
                      color: reportTab === t.key ? C.accentB : C.txts,
                      border: `1px solid ${reportTab === t.key ? 'rgba(59, 130, 246, 0.3)' : 'transparent'}`,
                      borderBottom: "none",
                      borderRadius: "8px 8px 0 0",
                      transition:"all .15s",
                      position: "relative",
                      top: "1px"
                    }}>{t.label}</button>
                  ))}
                </div>

                <div style={{ flex:1, overflowY:"auto" }}>
                  {reportTab === "pdf" && pdfUrl ? (
                    <iframe
                      src={pdfUrl + "#toolbar=1"}
                      title="PDF Report"
                      style={{ width:"100%", height:"100%", minHeight:600, border:"none", display:"block" }}
                    />
                  ) : (
                    <ReportRenderer text={report} />
                  )}
                </div>
              </Panel>

              <Panel cls="si" style={{ display:"flex", flexDirection:"column" }}>
                <div style={panelHeader}>
                  <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#34d399" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
                    <span style={{ fontFamily:"'Inter',sans-serif", fontWeight:600, fontSize:".95rem", color:C.txts, textTransform: "uppercase", letterSpacing: "1px" }}>RAG Assistant</span>
                  </div>
                </div>

                <div style={{ flex:1, overflowY:"auto", padding:"20px 24px", display:"flex", flexDirection:"column", gap:16 }}>
                  {history.length === 0 && (
                    <p style={{ color:C.txtt, fontSize:".85rem", lineHeight:1.7, marginTop:2, textAlign: "center" }}>
                      Ask anything about the findings, measurements, or clinical significance.
                    </p>
                  )}

                  {history.map((item, i) => (
                    <div key={i} style={{ display:"flex", flexDirection:"column", gap:10, animation:"chatIn .22s ease both" }}>
                      <div style={{ alignSelf:"flex-end", maxWidth:"88%", background:C.accent, borderRadius:"12px 12px 0px 12px", padding:"12px 16px", fontSize:".85rem", fontWeight:500, color:"#fff", lineHeight:1.55 }}>
                        {item.q}
                      </div>
                      <div style={{ alignSelf:"flex-start", maxWidth:"94%", display:"flex", gap:12, alignItems:"flex-start" }}>
                        <div style={{ width:28, height:28, borderRadius:"50%", background:C.accentDim, border:`1px solid rgba(59, 130, 246, 0.3)`, display:"flex", alignItems:"center", justifyContent:"center", fontSize:".8rem", flexShrink:0, marginTop:2 }}>🧠</div>
                        <div style={{ background:C.cardB, border:`1px solid ${C.border}`, borderRadius:"0px 12px 12px 12px", padding:"12px 16px", fontSize:".85rem", color:C.txts, lineHeight:1.68 }}>
                          <ReactMarkdown components={{
                            ul: ({children}) => <ul style={{margin:"6px 0", paddingLeft:"1.2em"}}>{children}</ul>,
                            ol: ({children}) => <ol style={{margin:"6px 0", paddingLeft:"1.2em"}}>{children}</ol>,
                            li: ({children}) => <li style={{marginBottom:"4px"}}>{children}</li>,
                            p: ({children}) => <p style={{margin:"4px 0"}}>{children}</p>,
                            strong: ({children}) => <strong style={{fontWeight:600, color:"#e2e8f0"}}>{children}</strong>,
                          }}>{item.a}</ReactMarkdown>
                        </div>
                      </div>
                    </div>
                  ))}

                  {asking && (
                    <div style={{ display:"flex", alignItems:"center", gap:8, color:C.txtt, fontSize:".85rem", padding: "8px 0" }}>
                      <Spinner size={14} color={C.accentB} /> Processing query...
                    </div>
                  )}
                  <div ref={chatEndRef} />
                </div>

                <div style={{ padding:"16px 24px 24px", background: "#0d131b", borderTop:`1px solid ${C.border}`, flexShrink:0, display:"flex", flexDirection:"column", gap:12 }}>
                  {history.length === 0 && (
                    <div style={{ display:"flex", flexWrap:"nowrap", overflowX: "auto", gap:8, paddingBottom: 4 }}>
                      {["Summarize Findings", "Measurements?", "Draft Patient Note"].map(s => (
                        <div key={s} onClick={() => setQuestion(s)}
                          style={{ whiteSpace: "nowrap", background:C.card, border:`1px solid ${C.border}`, borderRadius:50, padding:"6px 14px", fontSize:".75rem", color:C.txts, cursor:"pointer", transition:"all .14s", fontWeight:500 }}
                          onMouseEnter={e => { e.currentTarget.style.background = C.border; e.currentTarget.style.color = C.txt; }}
                          onMouseLeave={e => { e.currentTarget.style.background = C.card;   e.currentTarget.style.color = C.txts; }}
                        >{s}</div>
                      ))}
                    </div>
                  )}

                  <div style={{ display:"flex", gap:10, alignItems:"center", position: "relative" }}>
                    <input
                      value={question} onChange={e => setQuestion(e.target.value)}
                      onKeyDown={e => e.key === "Enter" && ask()}
                      placeholder="Ask about the report…"
                      disabled={asking}
                      style={{ flex:1, padding:"14px 16px", paddingRight: "48px", borderRadius:12, border:`1px solid ${C.border}`, background:C.cardB, fontSize:".9rem", color:C.txt, outline:"none", transition:"border-color .15s, box-shadow .15s" }}
                      onFocus={e => { e.target.style.borderColor = C.accentB; e.target.style.boxShadow = "0 0 0 3px rgba(37,99,235,.14)"; }}
                      onBlur={e  => { e.target.style.borderColor = C.border;  e.target.style.boxShadow = "none"; }}
                    />
                    <button onClick={ask} disabled={asking || !question.trim()}
                      style={{ position: "absolute", right: 8, width:36, height:36, borderRadius:8, border:"none", flexShrink:0, cursor:(asking||!question.trim())?"not-allowed":"pointer", background:(asking||!question.trim())?"transparent":C.accent, color:(asking||!question.trim())?C.txtt:"#fff", display:"flex", alignItems:"center", justifyContent:"center", transition:"background .15s" }}>
                      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/></svg>
                    </button>
                  </div>
                </div>
              </Panel>

            </div>
          )}
        </main>
      </div>
    </>
  );
}