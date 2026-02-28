import { useState, useEffect, useRef } from "react";

const API_BASE = "http://localhost:8000";

const G = `
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Syne:wght@600;700;800&display=swap');
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: #060e1e; font-family: 'Inter', sans-serif; color: #e2eaf8; -webkit-font-smoothing: antialiased; }
  ::-webkit-scrollbar { width: 4px; }
  ::-webkit-scrollbar-thumb { background: #1e3a6a; border-radius: 4px; }
  input, textarea, button { font-family: inherit; }
  @keyframes spin   { to { transform: rotate(360deg) } }
  @keyframes pulse  { 0%,100%{opacity:1} 50%{opacity:.2} }
  @keyframes fadeUp { from{opacity:0;transform:translateY(14px)} to{opacity:1;transform:translateY(0)} }
  @keyframes slideIn{ from{opacity:0;transform:translateX(16px)} to{opacity:1;transform:translateX(0)} }
  @keyframes glow   { 0%,100%{box-shadow:0 0 16px #2563eb44} 50%{box-shadow:0 0 32px #2563eb88} }
  @keyframes scan   { 0%{top:-6px} 100%{top:100%} }
  @keyframes chatIn { from{opacity:0;transform:translateY(6px)} to{opacity:1;transform:translateY(0)} }
  .fu  { animation: fadeUp  .4s ease both; }
  .fu2 { animation: fadeUp  .4s .08s ease both; }
  .si  { animation: slideIn .38s ease both; }
`;

const C = {
  bg0: "#060e1e", bg1: "#0b1628", bg2: "#0f1e38", bg3: "#132244",
  border: "#1a3358", borderB: "#1f3d6a",
  accent: "#2563eb", accentB: "#3b7ff5", accentL: "#1a3980",
  txt: "#e2eaf8", txts: "#8ba4c8", txtt: "#4a6490",
  ok: "#10b981", okBg: "#052e20",
  warn: "#f59e0b", warnBg: "#2d1f05", warnBdr: "#78450a",
  err: "#ef4444", errBg: "#2a0a0a",
  grid: "rgba(30,60,110,.18)",
};

const Spinner = ({ size = 16, color = C.accentB }) => (
  <span style={{ display:"inline-block", width:size, height:size, border:`2px solid ${color}28`, borderTopColor:color, borderRadius:"50%", animation:"spin .7s linear infinite", flexShrink:0 }} />
);

function Btn({ children, onClick, disabled, loading, ghost, full, sm }) {
  const [h, setH] = useState(false);
  const pad = sm ? ".4rem .9rem" : ".65rem 1.5rem";
  const fs  = sm ? ".77rem" : ".86rem";
  return (
    <button onClick={onClick} disabled={disabled||loading}
      onMouseEnter={()=>setH(true)} onMouseLeave={()=>setH(false)}
      style={{
        width: full?"100%":"auto", display:"inline-flex", alignItems:"center", justifyContent:"center", gap:7,
        padding:pad, borderRadius:9, fontWeight:600, fontSize:fs,
        border: ghost?`1px solid ${h?C.accentB:C.border}`:"none",
        cursor:(disabled||loading)?"not-allowed":"pointer",
        background: ghost?(h?"rgba(59,127,245,.12)":"transparent"):(disabled||loading)?"#0f1e38":h?C.accentB:C.accent,
        color: ghost?(h?C.accentB:C.txts):(disabled||loading)?C.txtt:"#fff",
        boxShadow:!ghost&&!(disabled||loading)?(h?"0 0 24px #2563eb66":"0 0 12px #2563eb44"):"none",
        transform:!ghost&&!(disabled||loading)&&h?"translateY(-1px)":"none",
        transition:"all .16s ease",
      }}>
      {loading && <Spinner size={13} color={ghost?C.accentB:"#fff"} />}
      {children}
    </button>
  );
}

const Card = ({ children, style={}, cls="" }) => (
  <div className={cls} style={{
    background:`linear-gradient(145deg,${C.bg2},${C.bg1})`,
    border:`1px solid ${C.border}`,
    borderRadius:16,
    boxShadow:"0 4px 24px rgba(0,0,0,.4), inset 0 1px 0 rgba(255,255,255,.04)",
    overflow:"hidden",
    ...style,
  }}>{children}</div>
);

const AccentBar = ({ dir = "left" }) => (
  <div style={{ height:3, background: dir==="left" ? `linear-gradient(90deg,${C.accent},${C.accentB},transparent)` : `linear-gradient(90deg,transparent,${C.accentB},${C.accent})` }} />
);

const SectionLabel = ({ children }) => (
  <p style={{ fontFamily:"'Syne',sans-serif", fontSize:".67rem", fontWeight:700, letterSpacing:"1.1px", textTransform:"uppercase", color:C.txtt, marginBottom:12 }}>{children}</p>
);

const REQUIRED = ["t1","t1ce","t2","flair"];
const ALIASES  = { t1n:"t1", t1c:"t1ce", t2w:"t2", t2f:"flair" };
function detectMods(files) {
  const found = new Set();
  for (const f of files) {
    const stem   = f.name.toLowerCase().replace(/\.nii\.gz$|\.nii$/, "");
    const parts  = stem.split(/[_-]/);
    const suffix = parts[parts.length-1];
    found.add(ALIASES[suffix]??suffix);
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

  const phase = !jobId?"upload":report?"done":"processing";

  useEffect(()=>{
    if (phase!=="processing") return;
    let timer;
    const poll = async ()=>{
      try {
        const r=await fetch(`${API_BASE}/api/status/${jobId}`);
        const d=await r.json();
        setProgress(Math.min(d.progress_pct??0,100));
        setStage(d.stage??"");
        if (d.status==="done") {
          const rr=await fetch(`${API_BASE}/api/report/${jobId}`);
          setReport(await rr.text());
          try {
            const pr=await fetch(`${API_BASE}/api/report/${jobId}/pdf`);
            if (pr.ok) {
              const blob=await pr.blob();
              setPdfUrl(URL.createObjectURL(blob));
            }
          } catch {}
          return;
        }
        if (d.status==="error") { setUploadErr(d.error_message||"Pipeline error."); setJobId(null); return; }
      } catch {}
      timer=setTimeout(poll,4000);
    };
    poll();
    return ()=>clearTimeout(timer);
  },[phase,jobId]);

  useEffect(()=>{ chatEndRef.current?.scrollIntoView({behavior:"smooth"}); },[history,asking]);

  const handleFiles = (list)=>{ setFiles(Array.from(list)); setUploadErr(""); };

  const submit = async ()=>{
    setUploading(true); setUploadErr("");
    try {
      const fd=new FormData();
      fd.append("case_id",caseId.trim());
      for (const f of files) fd.append("files",f);
      const r=await fetch(`${API_BASE}/api/analyze`,{method:"POST",body:fd});
      if (!r.ok) throw new Error((await r.json()).detail??"Upload failed");
      const {job_id}=await r.json();
      setJobId(job_id); setProgress(0);
    } catch(e){ setUploadErr(e.message); }
    finally { setUploading(false); }
  };

  const ask = async ()=>{
    if (!question.trim()||asking) return;
    const q=question.trim(); setQuestion(""); setAsking(true);
    try {
      const r=await fetch(`${API_BASE}/api/chat/${jobId}`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({question:q})});
      const d=await r.json();
      setHistory(h=>[...h,{q,a:d.answer||"No answer returned."}]);
    } catch(e){ setHistory(h=>[...h,{q,a:"Error: "+e.message}]); }
    finally { setAsking(false); }
  };

  const reset = ()=>{
    if (pdfUrl) URL.revokeObjectURL(pdfUrl);
    setFiles([]); setCaseId(""); setJobId(null); setReport(""); setPdfUrl("");
    setProgress(0); setStage(""); setHistory([]); setQuestion(""); setUploadErr(""); setReportTab("pdf");
  };

  const found      = detectMods(files);
  const missing    = REQUIRED.filter(m=>!found.has(m));
  const allPresent = files.length>0 && missing.length===0;
  const ready      = allPresent && caseId.trim() && !uploading;

  return (
    <>
      <style>{G}</style>

      {/* Dot grid bg */}
      <div style={{position:"fixed",inset:0,zIndex:0,backgroundImage:`radial-gradient(${C.grid} 1px,transparent 1px)`,backgroundSize:"28px 28px",pointerEvents:"none"}} />
      {/* Top glow */}
      <div style={{position:"fixed",top:-140,left:"50%",transform:"translateX(-50%)",width:800,height:320,background:"radial-gradient(ellipse,#1e3a8a3a 0%,transparent 70%)",zIndex:0,pointerEvents:"none"}} />

      <div style={{position:"relative",zIndex:1,minHeight:"100vh",display:"flex",flexDirection:"column"}}>

        {/* ══ HEADER ══ */}
        <header style={{borderBottom:`1px solid ${C.border}`,background:"rgba(6,14,30,.85)",backdropFilter:"blur(14px)",padding:"0 40px",height:62,display:"flex",alignItems:"center",justifyContent:"space-between",flexShrink:0,position:"sticky",top:0,zIndex:100}}>
          <div style={{display:"flex",alignItems:"center",gap:12}}>
            <div style={{width:36,height:36,borderRadius:10,background:"linear-gradient(135deg,#1e40af,#2563eb)",display:"flex",alignItems:"center",justifyContent:"center",boxShadow:"0 0 18px #2563eb55",fontSize:"1rem"}}>🧠</div>
            <div>
              <span style={{fontFamily:"'Syne',sans-serif",fontWeight:800,fontSize:".95rem",color:"#fff",letterSpacing:"-.2px"}}>AI-Powered Brain MRI Assistant</span>
            </div>
          </div>

          <div style={{display:"flex",alignItems:"center",gap:10}}>
            {phase==="processing" && (
              <div style={{display:"flex",alignItems:"center",gap:8,background:C.warnBg,border:`1px solid ${C.warnBdr}`,borderRadius:50,padding:"5px 14px"}}>
                <span style={{width:7,height:7,borderRadius:"50%",background:C.warn,display:"inline-block",animation:"pulse 1.2s ease infinite"}} />
                <span style={{color:C.warn,fontSize:".74rem",fontWeight:600}}>Analysis running</span>
              </div>
            )}
            {phase==="done" && (
              <div style={{display:"flex",alignItems:"center",gap:8,background:C.okBg,border:"1px solid #0d6640",borderRadius:50,padding:"5px 14px"}}>
                <span style={{color:C.ok,fontSize:".74rem",fontWeight:600}}>✓ Report ready</span>
              </div>
            )}
            {phase==="done" && <Btn ghost sm onClick={reset}>↺ New analysis</Btn>}
          </div>
        </header>

        {/* ══ MAIN ══ */}
        <main style={{flex:1,padding:"30px 40px 48px",display:"flex",flexDirection:"column",gap:20}}>

          {/* ── UPLOAD ROW ── */}
          <section style={{opacity:phase==="done"?.28:1,pointerEvents:phase==="done"?"none":"auto",transition:"opacity .5s"}}>
            <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:16}}>

              {/* Drop zone */}
              <Card cls="fu">
                <div style={{padding:"24px 26px"}}>
                  <SectionLabel>MRI Sequences</SectionLabel>
                  <div
                    onClick={()=>fileRef.current?.click()}
                    onDragOver={e=>{e.preventDefault();setDragging(true);}}
                    onDragLeave={()=>setDragging(false)}
                    onDrop={e=>{e.preventDefault();setDragging(false);handleFiles(e.dataTransfer.files);}}
                    style={{
                      background:dragging?"rgba(59,127,245,.08)":"rgba(255,255,255,.02)",
                      border:`2px dashed ${dragging?C.accentB:C.border}`,
                      borderRadius:12,padding:files.length?"16px 18px":"28px 18px",
                      textAlign:"center",cursor:"pointer",transition:"all .2s",
                      position:"relative",overflow:"hidden",
                    }}
                  >
                    {dragging && <div style={{position:"absolute",left:0,right:0,height:2,background:`linear-gradient(90deg,transparent,${C.accentB},transparent)`,animation:"scan 1s linear infinite"}} />}
                    {files.length===0 ? (
                      <>
                        <div style={{fontSize:"1.5rem",marginBottom:8,opacity:.25}}>↑</div>
                        <p style={{color:C.txts,fontWeight:500,fontSize:".87rem"}}>
                          Drop <code style={{background:"rgba(59,127,245,.2)",color:C.accentB,padding:"1px 6px",borderRadius:4,fontSize:".78rem"}}>.nii.gz</code> files or{" "}
                          <span style={{color:C.accentB,fontWeight:600}}>browse</span>
                        </p>
                        <p style={{color:C.txtt,fontSize:".73rem",marginTop:5}}>T1 · T1CE · T2 · FLAIR required</p>
                      </>
                    ) : (
                      <div style={{textAlign:"left"}}>
                        <p style={{fontSize:".79rem",fontWeight:600,color:C.txts,marginBottom:8}}>
                          {files.length} file{files.length>1?"s":""} selected —{" "}
                          <span onClick={e=>{e.stopPropagation();setFiles([]);}} style={{color:C.accentB,cursor:"pointer"}}>clear</span>
                        </p>
                        <div style={{display:"flex",flexWrap:"wrap",gap:5}}>
                          {files.map(f=>(
                            <span key={f.name} style={{background:"rgba(59,127,245,.15)",color:C.accentB,border:`1px solid ${C.accentL}`,borderRadius:6,padding:"2px 9px",fontSize:".72rem",fontWeight:600}}>{f.name}</span>
                          ))}
                        </div>
                      </div>
                    )}
                    <input ref={fileRef} type="file" accept=".nii.gz,.nii" multiple style={{display:"none"}} onChange={e=>handleFiles(e.target.files)} />
                  </div>

                  {files.length>0 && (
                    <div style={{display:"flex",gap:7,marginTop:12}}>
                      {REQUIRED.map(mod=>{
                        const ok=found.has(mod);
                        return <div key={mod} style={{flex:1,textAlign:"center",padding:"6px 2px",borderRadius:8,fontSize:".72rem",fontWeight:700,background:ok?C.okBg:C.warnBg,color:ok?C.ok:C.warn,border:`1px solid ${ok?"#0d6640":C.warnBdr}`,transition:"all .25s"}}>{ok?"✓":"✗"} {mod.toUpperCase()}</div>;
                      })}
                    </div>
                  )}
                </div>
                <AccentBar />
              </Card>

              {/* Case ID + submit */}
              <Card cls="fu2" style={{display:"flex",flexDirection:"column"}}>
                <div style={{padding:"24px 26px",flex:1,display:"flex",flexDirection:"column",gap:16}}>
                  <div>
                    <SectionLabel>Case Details</SectionLabel>
                    <label style={{display:"block",fontSize:".79rem",fontWeight:500,color:C.txts,marginBottom:7}}>Patient / Case ID</label>
                    <input
                      value={caseId} onChange={e=>setCaseId(e.target.value)}
                      placeholder="e.g. BraTS-GLI-00003-000"
                      style={{width:"100%",padding:"11px 14px",borderRadius:9,border:`1px solid ${C.border}`,background:"rgba(255,255,255,.04)",fontSize:".87rem",color:C.txt,outline:"none",transition:"border-color .18s, box-shadow .18s"}}
                      onFocus={e=>{e.target.style.borderColor=C.accentB;e.target.style.boxShadow="0 0 0 3px rgba(37,99,235,.18)";}}
                      onBlur={e=>{e.target.style.borderColor=C.border;e.target.style.boxShadow="none";}}
                    />
                  </div>

                  {/* 2 info tiles */}
                  <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:8}}>
                    {[
                      {label:"Segmentation",val:"nnUNet v2"},
                      {label:"Processing",   val:"3 – 5 min"},
                    ].map(({label,val})=>(
                      <div key={label} style={{background:"rgba(255,255,255,.03)",border:`1px solid ${C.border}`,borderRadius:9,padding:"10px 12px"}}>
                        <p style={{fontSize:".66rem",fontWeight:600,letterSpacing:".5px",textTransform:"uppercase",color:C.txtt,marginBottom:3}}>{label}</p>
                        <p style={{fontSize:".84rem",fontWeight:700,color:C.txt}}>{val}</p>
                      </div>
                    ))}
                  </div>

                  {uploadErr && <div style={{background:C.errBg,border:"1px solid #7f1d1d",borderRadius:9,padding:"9px 13px",fontSize:".79rem",color:C.err}}>⚠ {uploadErr}</div>}

                  <div style={{marginTop:"auto"}}>
                    <Btn full onClick={submit} disabled={!ready} loading={uploading}>
                      {uploading?"Uploading…":"Run Analysis →"}
                    </Btn>
                  </div>
                </div>
                <AccentBar dir="right" />
              </Card>
            </div>
          </section>

          {/* ── PROGRESS ── */}
          {phase==="processing" && (
            <Card cls="fu" style={{padding:"22px 28px"}}>
              <div style={{display:"flex",alignItems:"center",gap:18}}>
                <div style={{animation:"glow 2s ease infinite",borderRadius:"50%",flexShrink:0}}>
                  <Spinner size={24} color={C.accentB} />
                </div>
                <div style={{flex:1}}>
                  <div style={{display:"flex",justifyContent:"space-between",marginBottom:8}}>
                    <p style={{fontWeight:600,fontSize:".9rem",color:C.txt}}>
                      {stage?stage.charAt(0).toUpperCase()+stage.slice(1)+"…":"Processing…"}
                    </p>
                    <p style={{fontWeight:700,fontSize:".9rem",color:C.accentB}}>{progress}%</p>
                  </div>
                  <div style={{height:6,background:"rgba(59,127,245,.12)",borderRadius:50,overflow:"hidden"}}>
                    <div style={{height:"100%",width:`${progress}%`,background:`linear-gradient(90deg,${C.accent},${C.accentB})`,borderRadius:50,transition:"width .6s ease",boxShadow:`0 0 10px ${C.accent}`}} />
                  </div>
                  <p style={{fontSize:".72rem",color:C.txtt,marginTop:7}}>Analysis typically completes in 3 – 5 minutes. Keep this tab open.</p>
                </div>
              </div>
            </Card>
          )}

          {/* ── REPORT + CHAT ── */}
          {phase==="done" && (
            <div style={{display:"grid",gridTemplateColumns:"1fr 380px",gap:16,animation:"fadeUp .4s ease both",flex:1}}>

              {/* Report */}
              <Card style={{display:"flex",flexDirection:"column"}}>
                <div style={{padding:"18px 24px 0",display:"flex",alignItems:"center",justifyContent:"space-between",flexShrink:0}}>
                  <div style={{display:"flex",alignItems:"center",gap:10}}>
                    <div style={{width:8,height:8,borderRadius:"50%",background:C.ok,boxShadow:`0 0 8px ${C.ok}`}} />
                    <span style={{fontFamily:"'Syne',sans-serif",fontWeight:700,fontSize:".9rem",color:C.txt}}>Radiology Report</span>
                    <span style={{color:C.txtt,fontSize:".78rem"}}>— {caseId}</span>
                  </div>
                  <div style={{display:"flex",gap:6}}>
                    {pdfUrl && <Btn ghost sm onClick={()=>{
                      const a=document.createElement("a");a.href=pdfUrl;a.download=`${caseId}.pdf`;a.click();
                    }}>⬇ PDF</Btn>}
                    <Btn ghost sm onClick={()=>{
                      const b=new Blob([report],{type:"text/plain"});
                      const a=document.createElement("a");a.href=URL.createObjectURL(b);a.download=`${caseId}.txt`;a.click();
                    }}>⬇ Text</Btn>
                  </div>
                </div>

                {/* Tab bar */}
                <div style={{display:"flex",gap:0,padding:"12px 24px 0",flexShrink:0}}>
                  {[{key:"pdf",label:"📄 PDF Report"},{key:"text",label:"📝 Text Report"}].map(t=>(
                    <button key={t.key} onClick={()=>setReportTab(t.key)} style={{
                      padding:"8px 18px",fontSize:".78rem",fontWeight:600,cursor:"pointer",
                      background:reportTab===t.key?"rgba(59,127,245,.15)":"transparent",
                      color:reportTab===t.key?C.accentB:C.txts,
                      border:`1px solid ${reportTab===t.key?C.accentL:C.border}`,
                      borderRadius:t.key==="pdf"?"8px 0 0 8px":"0 8px 8px 0",
                      transition:"all .15s",
                    }}>{t.label}</button>
                  ))}
                </div>

                <div style={{flex:1,padding:"16px 24px 24px"}}>
                  {reportTab==="pdf" && pdfUrl ? (
                    <iframe
                      src={pdfUrl+"#toolbar=1"}
                      title="PDF Report"
                      style={{width:"100%",height:"100%",minHeight:540,border:`1px solid ${C.border}`,borderRadius:10,background:"#fff"}}
                    />
                  ) : (
                    <textarea
                      readOnly value={report}
                      style={{width:"100%",height:"100%",minHeight:540,resize:"none",border:`1px solid ${C.border}`,borderRadius:10,padding:"16px 18px",fontSize:".82rem",lineHeight:1.85,background:"rgba(255,255,255,.025)",color:C.txt,outline:"none",fontFamily:"inherit",transition:"border-color .18s"}}
                      onFocus={e=>e.target.style.borderColor=C.accentL}
                      onBlur={e=>e.target.style.borderColor=C.border}
                    />
                  )}
                </div>
                <AccentBar />
              </Card>

              {/* Chat */}
              <Card cls="si" style={{display:"flex",flexDirection:"column"}}>
                <div style={{padding:"18px 22px 14px",borderBottom:`1px solid ${C.border}`,flexShrink:0}}>
                  <div style={{display:"flex",alignItems:"center",gap:8}}>
                    <div style={{width:8,height:8,borderRadius:"50%",background:C.accentB,boxShadow:`0 0 8px ${C.accentB}`,animation:"pulse 2s ease infinite"}} />
                    <span style={{fontFamily:"'Syne',sans-serif",fontWeight:700,fontSize:".9rem",color:C.txt}}>Clinical Q&A</span>
                  </div>
                  <p style={{fontSize:".72rem",color:C.txtt,marginTop:5}}>Powered by RAG · Grounded in this report</p>
                </div>

                <div style={{flex:1,overflowY:"auto",padding:"16px 18px",display:"flex",flexDirection:"column",gap:12}}>
                  {history.length===0 && (
                    <div style={{marginTop:4}}>
                      <p style={{color:C.txtt,fontSize:".8rem",lineHeight:1.7,marginBottom:14}}>Ask anything about the findings.</p>
                      {["What regions show enhancement?","Is there midline shift?","What is the tumor volume?"].map(s=>(
                        <div key={s} onClick={()=>setQuestion(s)}
                          style={{background:"rgba(59,127,245,.07)",border:`1px solid ${C.border}`,borderRadius:8,padding:"8px 12px",fontSize:".78rem",color:C.txts,cursor:"pointer",marginBottom:6,transition:"all .15s"}}
                          onMouseEnter={e=>{e.currentTarget.style.borderColor=C.accentL;e.currentTarget.style.color=C.accentB;}}
                          onMouseLeave={e=>{e.currentTarget.style.borderColor=C.border;e.currentTarget.style.color=C.txts;}}
                        >{s}</div>
                      ))}
                    </div>
                  )}

                  {history.map((item,i)=>(
                    <div key={i} style={{display:"flex",flexDirection:"column",gap:7,animation:"chatIn .25s ease both"}}>
                      <div style={{alignSelf:"flex-end",maxWidth:"88%",background:"linear-gradient(135deg,#1a3980,rgba(37,99,235,.28))",border:`1px solid ${C.accentL}`,borderRadius:"12px 12px 3px 12px",padding:"9px 13px",fontSize:".81rem",fontWeight:600,color:"#c7d9ff"}}>
                        {item.q}
                      </div>
                      <div style={{alignSelf:"flex-start",maxWidth:"93%",background:"rgba(255,255,255,.04)",border:`1px solid ${C.border}`,borderRadius:"3px 12px 12px 12px",padding:"9px 13px",fontSize:".81rem",color:C.txts,lineHeight:1.68}}>
                        {item.a}
                      </div>
                    </div>
                  ))}

                  {asking && (
                    <div style={{display:"flex",alignItems:"center",gap:8,color:C.txtt,fontSize:".77rem"}}>
                      <Spinner size={12} color={C.accentB} /> Thinking…
                    </div>
                  )}
                  <div ref={chatEndRef} />
                </div>

                <div style={{padding:"14px 18px 20px",borderTop:`1px solid ${C.border}`,flexShrink:0}}>
                  <div style={{display:"flex",gap:8}}>
                    <input
                      value={question} onChange={e=>setQuestion(e.target.value)}
                      onKeyDown={e=>e.key==="Enter"&&ask()}
                      placeholder="Ask a question…"
                      disabled={asking}
                      style={{flex:1,padding:"10px 13px",borderRadius:9,border:`1px solid ${C.border}`,background:"rgba(255,255,255,.05)",fontSize:".83rem",color:C.txt,outline:"none",transition:"border-color .18s, box-shadow .18s"}}
                      onFocus={e=>{e.target.style.borderColor=C.accentB;e.target.style.boxShadow="0 0 0 3px rgba(37,99,235,.18)";}}
                      onBlur={e=>{e.target.style.borderColor=C.border;e.target.style.boxShadow="none";}}
                    />
                    <Btn onClick={ask} disabled={asking||!question.trim()} loading={asking}>→</Btn>
                  </div>
                </div>
                <AccentBar dir="right" />
              </Card>

            </div>
          )}
        </main>
      </div>
    </>
  );
}
