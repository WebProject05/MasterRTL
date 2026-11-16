"""MasterRTL Interactive EDA Web Studio.

Zero-dependency local web application providing an industry-grade GUI for chip design engineers:
- Live Verilog code viewer & editor
- One-click presynthesis PPA estimation engine
- Live visual gauges for Silicon Area, Timing (WNS/TNS), Dynamic & Static Power
- Real-time SOG graph complexity metrics & STA delay distribution
- Multi-foundry technology corner switcher (NanGate 45nm, TSMC 28nm)
- Live EDA console log with execution telemetry
"""

from __future__ import annotations

import argparse
import http.server
import json
import os
import pickle
import socketserver
import sys
import tempfile
import threading
import time
import urllib.parse
from pathlib import Path
from typing import Any

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.predict import (
    generate_prediction_plot,
    predict_verilog_file,
)
from src.models.master_rtl import MasterRTL

_LOADED_MODEL: MasterRTL | None = None


def get_model() -> MasterRTL:
    global _LOADED_MODEL
    if _LOADED_MODEL is None:
        ckpt_path = PROJECT_ROOT / "models" / "master_rtl_checkpoint.pkl"
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Model checkpoint not found at {ckpt_path}. Run train_all.py first.")
        with ckpt_path.open("rb") as f:
            _LOADED_MODEL = pickle.load(f)
    return _LOADED_MODEL


STUDIO_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MasterRTL EDA Studio | Presynthesis PPA Workbench</title>
    <style>
        :root {
            --bg-dark: #0f172a;
            --bg-panel: #1e293b;
            --bg-sub: #334155;
            --accent: #38bdf8;
            --accent-green: #34d399;
            --accent-red: #f87171;
            --accent-amber: #fbbf24;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --border: #475569;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, monospace; }
        body { background: var(--bg-dark); color: var(--text-main); display: flex; flex-direction: column; height: 100vh; overflow: hidden; }

        /* Top Navigation Bar */
        .topbar {
            background: #090d16;
            border-bottom: 1px solid var(--border);
            padding: 10px 20px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        .logo-area { display: flex; align-items: center; gap: 12px; }
        .chip-icon {
            background: linear-gradient(135deg, #0284c7, #38bdf8);
            color: #0f172a;
            font-weight: 900;
            padding: 6px 10px;
            border-radius: 6px;
            font-size: 1rem;
            letter-spacing: 1px;
        }
        .logo-title { font-size: 1.15rem; font-weight: 700; color: #ffffff; letter-spacing: -0.3px; }
        .logo-subtitle { font-size: 0.75rem; color: var(--text-muted); }
        .tech-badge {
            background: rgba(56, 189, 248, 0.15);
            border: 1px solid rgba(56, 189, 248, 0.4);
            color: var(--accent);
            padding: 4px 10px;
            border-radius: 4px;
            font-size: 0.78rem;
            font-weight: 600;
        }

        /* Workspace Grid */
        .workspace { display: flex; flex: 1; overflow: hidden; }

        /* Left Sidebar: Design Navigator & Controls */
        .sidebar {
            width: 320px;
            background: var(--bg-panel);
            border-right: 1px solid var(--border);
            display: flex;
            flex-direction: column;
            overflow-y: auto;
        }
        .sidebar-section { padding: 16px; border-bottom: 1px solid var(--border); }
        .section-label { font-size: 0.75rem; font-weight: 700; text-transform: uppercase; color: var(--text-muted); margin-bottom: 10px; letter-spacing: 0.5px; }

        select, input, button {
            width: 100%;
            background: var(--bg-dark);
            border: 1px solid var(--border);
            color: var(--text-main);
            padding: 9px 12px;
            border-radius: 6px;
            font-size: 0.88rem;
            margin-bottom: 10px;
            outline: none;
        }
        select:focus, input:focus { border-color: var(--accent); }

        .btn-run {
            background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%);
            color: white;
            font-weight: 700;
            border: none;
            padding: 12px;
            border-radius: 6px;
            cursor: pointer;
            box-shadow: 0 4px 12px rgba(2, 132, 199, 0.3);
            transition: all 0.2s;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
        }
        .btn-run:hover { filter: brightness(1.15); transform: translateY(-1px); }
        .btn-run:disabled { opacity: 0.5; cursor: not-allowed; }

        /* Main Workspace */
        .main-stage { flex: 1; display: flex; flex-direction: column; overflow: hidden; background: #0b1120; }

        /* KPI Banner */
        .kpi-row {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 12px;
            padding: 14px 20px;
            background: var(--bg-panel);
            border-bottom: 1px solid var(--border);
        }
        .metric-card {
            background: var(--bg-dark);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 12px 14px;
        }
        .metric-header { font-size: 0.72rem; text-transform: uppercase; font-weight: 600; color: var(--text-muted); }
        .metric-val { font-size: 1.45rem; font-weight: 800; color: #ffffff; margin: 3px 0; }
        .metric-desc { font-size: 0.75rem; color: var(--accent); }

        /* Center Content: Tabs */
        .tab-bar { display: flex; background: #090d16; border-bottom: 1px solid var(--border); padding: 0 16px; }
        .tab-item {
            padding: 10px 18px;
            font-size: 0.85rem;
            font-weight: 600;
            color: var(--text-muted);
            cursor: pointer;
            border-bottom: 2px solid transparent;
        }
        .tab-item.active { color: var(--accent); border-bottom-color: var(--accent); }

        .tab-view { flex: 1; overflow-y: auto; padding: 18px; display: none; }
        .tab-view.active { display: block; }

        /* Editor */
        .code-container { height: 100%; display: flex; flex-direction: column; }
        textarea.editor {
            flex: 1;
            background: #050811;
            color: #38bdf8;
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 14px;
            font-family: 'Consolas', 'Courier New', monospace;
            font-size: 0.88rem;
            line-height: 1.5;
            resize: none;
            outline: none;
            width: 100%;
            height: 380px;
        }

        /* Image Display */
        .plot-container { text-align: center; }
        .plot-container img { max-width: 100%; height: auto; border-radius: 8px; border: 1px solid var(--border); box-shadow: 0 4px 16px rgba(0,0,0,0.5); }

        /* Bottom Console Log */
        .console-drawer {
            height: 160px;
            background: #050811;
            border-top: 1px solid var(--border);
            padding: 10px 16px;
            overflow-y: auto;
            font-family: 'Consolas', monospace;
            font-size: 0.80rem;
        }
        .console-line { margin: 3px 0; color: #cbd5e1; }
        .console-line.info { color: #38bdf8; }
        .console-line.success { color: #34d399; }
        .console-line.warn { color: #fbbf24; }
        .console-line.error { color: #f87171; }
    </style>
</head>
<body>

    <!-- Header -->
    <header class="topbar">
        <div class="logo-area">
            <div class="chip-icon">RTL</div>
            <div>
                <div class="logo-title">MasterRTL EDA Studio</div>
                <div class="logo-subtitle">Presynthesis PPA Estimation & Technology Transfer Engine</div>
            </div>
        </div>
        <div style="display: flex; gap: 10px; align-items: center;">
            <div class="tech-badge" id="current-node-badge">NanGate 45nm @ 1.0 GHz</div>
        </div>
    </header>

    <!-- Main Workspace -->
    <div class="workspace">
        
        <!-- Sidebar Controls -->
        <aside class="sidebar">
            <div class="sidebar-section">
                <div class="section-label">1. Select RTL Benchmark</div>
                <select id="design-select" onchange="loadSelectedDesign()">
                    <option value="">-- Choose Benchmark Design --</option>
                </select>
                <div style="font-size: 0.75rem; color: var(--text-muted); margin-top: -6px;">Or paste custom Verilog into editor tab</div>
            </div>

            <div class="sidebar-section">
                <div class="section-label">2. Synthesis & Technology Corner</div>
                <label style="font-size: 0.78rem; color: var(--text-muted); display: block; margin-bottom: 4px;">Foundry Node Scaling</label>
                <select id="tech-select">
                    <option value="">NanGate 45nm (Baseline Standard Cell)</option>
                    <option value="TSMC_28nm_TYP">TSMC 28nm (Typical Corner)</option>
                    <option value="TSMC_28nm_MIN">TSMC 28nm (Fast / Min Corner)</option>
                    <option value="TSMC_28nm_MAX">TSMC 28nm (Slow / Max Corner)</option>
                </select>

                <label style="font-size: 0.78rem; color: var(--text-muted); display: block; margin-bottom: 4px;">Clock Period (ns)</label>
                <input type="number" id="clk-input" value="1.0" step="0.1" min="0.2">
            </div>

            <div class="sidebar-section" style="flex: 1;">
                <button class="btn-run" id="btn-run" onclick="runAnalysis()">
                    <span>▶ Run Presynthesis PPA</span>
                </button>
            </div>
        </aside>

        <!-- Main Display -->
        <main class="main-stage">
            
            <!-- KPI Cards -->
            <div class="kpi-row">
                <div class="metric-card">
                    <div class="metric-header">Total Silicon Area</div>
                    <div class="metric-val" id="val-area">--</div>
                    <div class="metric-desc" id="sub-area">Seq / Comb Partition</div>
                </div>
                <div class="metric-card">
                    <div class="metric-header">Worst Slack (WNS)</div>
                    <div class="metric-val" id="val-wns">--</div>
                    <div class="metric-desc" id="sub-wns">Clock Margin Status</div>
                </div>
                <div class="metric-card">
                    <div class="metric-header">Total Slack (TNS)</div>
                    <div class="metric-val" id="val-tns">--</div>
                    <div class="metric-desc" id="sub-tns">Endpoint Violations</div>
                </div>
                <div class="metric-card">
                    <div class="metric-header">Total Power</div>
                    <div class="metric-val" id="val-power">--</div>
                    <div class="metric-desc" id="sub-power">Dynamic & Static Leakage</div>
                </div>
            </div>

            <!-- Tab Navigation -->
            <div class="tab-bar">
                <div class="tab-item active" onclick="switchTab(event, 'view-code')">Verilog RTL Source</div>
                <div class="tab-item" onclick="switchTab(event, 'view-plot')">4-Panel Visual Dashboard</div>
                <div class="tab-item" onclick="switchTab(event, 'view-stats')">SOG Circuit Structural Topology</div>
            </div>

            <!-- TAB 1: Verilog Code -->
            <div id="view-code" class="tab-view active">
                <div class="code-container">
                    <textarea class="editor" id="code-editor" spellcheck="false" placeholder="// Select a design from the left or paste synthesizable Verilog code here..."></textarea>
                </div>
            </div>

            <!-- TAB 2: Visual Dashboard -->
            <div id="view-plot" class="tab-view">
                <div class="plot-container">
                    <div id="plot-placeholder" style="color: var(--text-muted); padding: 40px;">
                        Click '▶ Run Presynthesis PPA' to generate the 4-panel visual dashboard.
                    </div>
                    <img id="ppa-plot-img" style="display: none;" alt="PPA Visual Dashboard">
                </div>
            </div>

            <!-- TAB 3: SOG Graph Stats -->
            <div id="view-stats" class="tab-view">
                <div style="background: var(--bg-panel); border: 1px solid var(--border); border-radius: 8px; padding: 18px;">
                    <h3 style="font-size: 1rem; margin-bottom: 12px; color: var(--accent);">Canonical SOG Graph Topology Details</h3>
                    <div id="stats-details" style="font-size: 0.88rem; line-height: 1.8; color: #cbd5e1;">
                        No analysis run yet. Select a design and click 'Run Presynthesis PPA'.
                    </div>
                </div>
            </div>

            <!-- Live Console Drawer -->
            <div class="console-drawer" id="console-drawer">
                <div class="console-line info">[INIT] MasterRTL EDA Studio v2.4.0 ready. Core models loaded.</div>
                <div class="console-line">[INFO] Ready to elaborate and estimate presynthesis PPA.</div>
            </div>

        </main>
    </div>

    <script>
        function log(msg, type = '') {
            const drawer = document.getElementById('console-drawer');
            const line = document.createElement('div');
            line.className = 'console-line ' + type;
            const time = new Date().toLocaleTimeString();
            line.innerText = `[${time}] ${msg}`;
            drawer.appendChild(line);
            drawer.scrollTop = drawer.scrollHeight;
        }

        function switchTab(evt, tabId) {
            document.querySelectorAll('.tab-view').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-item').forEach(b => b.classList.remove('active'));
            document.getElementById(tabId).classList.add('active');
            evt.currentTarget.classList.add('active');
        }

        async function init() {
            try {
                const res = await fetch('/api/designs');
                const list = await res.json();
                const sel = document.getElementById('design-select');
                list.forEach(name => {
                    const opt = document.createElement('option');
                    opt.value = name;
                    opt.innerText = name;
                    sel.appendChild(opt);
                });
                log(`Loaded ${list.length} benchmark designs from data/raw_rtl.`, 'info');
                
                // Auto-select adder_ripple_8bit
                if (list.includes('adder_ripple_8bit.v')) {
                    sel.value = 'adder_ripple_8bit.v';
                    loadSelectedDesign();
                }
            } catch (e) {
                log(`Failed to list designs: ${e}`, 'error');
            }
        }

        async function loadSelectedDesign() {
            const name = document.getElementById('design-select').value;
            if (!name) return;
            log(`Fetching RTL source for ${name}...`);
            try {
                const res = await fetch(`/api/design?name=${encodeURIComponent(name)}`);
                const data = await res.json();
                document.getElementById('code-editor').value = data.code;
                log(`Loaded ${name} (${data.code.split('\\n').length} lines).`, 'success');
            } catch (e) {
                log(`Failed to load design: ${e}`, 'error');
            }
        }

        async function runAnalysis() {
            const btn = document.getElementById('btn-run');
            btn.disabled = true;
            btn.innerText = 'Analyzing...';
            
            const designName = document.getElementById('design-select').value || 'custom_design.v';
            const code = document.getElementById('code-editor').value;
            const tech = document.getElementById('tech-select').value;
            const clk = parseFloat(document.getElementById('clk-input').value) || 1.0;

            log(`[SYNTH] Elaborating ${designName} into Simple Operator Graph (SOG)...`, 'info');

            try {
                const res = await fetch('/api/analyze', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        design_name: designName,
                        code: code,
                        clk_period: clk,
                        target_tech: tech,
                        generate_plot: true
                    })
                });

                const data = await res.json();
                if (data.error) {
                    log(`[ERROR] ${data.error}`, 'error');
                    btn.disabled = false;
                    btn.innerText = '▶ Run Presynthesis PPA';
                    return;
                }

                // Update KPIs
                const pred = data.prediction;
                const stats = data.stats;
                const wns = pred.wns;
                const area = pred.area_um2;
                const pwr = pred.power_mW;
                const tns = pred.tns;

                document.getElementById('val-area').innerText = `${area.toFixed(1)} μm²`;
                document.getElementById('sub-area').innerText = `Seq: ${(stats.num_registers*4.522).toFixed(1)} | Comb: ${(area - stats.num_registers*4.522).toFixed(1)}`;

                document.getElementById('val-wns').innerText = `${wns.toFixed(3)} ns`;
                document.getElementById('val-wns').style.color = wns >= 0 ? 'var(--accent-green)' : 'var(--accent-red)';
                document.getElementById('sub-wns').innerText = wns >= 0 ? 'TIMING MET (PASS)' : `VIOLATED by ${Math.abs(wns).toFixed(3)} ns`;

                document.getElementById('val-tns').innerText = `${tns.toFixed(3)} ns`;
                document.getElementById('val-power').innerText = `${pwr.toFixed(3)} mW`;

                if (data.transfer) {
                    log(`[TRANSFER] Scaled to ${data.transfer.target_technology}: Area=${data.transfer.transferred_area.toFixed(1)} μm², Power=${data.transfer.transferred_power.toFixed(3)} mW`, 'info');
                }

                log(`[SUCCESS] Presynthesis PPA generated in ${data.elapsed_seconds}s. Total Area: ${area.toFixed(1)} μm² | WNS: ${wns.toFixed(3)} ns | Power: ${pwr.toFixed(3)} mW`, 'success');

                // Update Plot
                if (data.plot_url) {
                    const img = document.getElementById('ppa-plot-img');
                    img.src = data.plot_url + '?t=' + new Date().getTime();
                    img.style.display = 'block';
                    document.getElementById('plot-placeholder').style.display = 'none';
                    log(`[PLOT] 4-panel visual PPA dashboard rendered.`, 'info');
                }

                // Update Stats Details Tab
                document.getElementById('stats-details').innerHTML = `
                    <strong>Design Name:</strong> ${data.top_module}<br>
                    <strong>Total 1-bit Nodes:</strong> ${stats.total_nodes}<br>
                    <strong>Registers (DFF):</strong> ${stats.num_registers} (${(stats.num_registers * 4.522).toFixed(2)} μm² sequential cell area)<br>
                    <strong>Combinational Logic:</strong> ${stats.num_comb_nodes} 1-bit gates (AND: ${stats.num_and}, OR: ${stats.num_or}, XOR: ${stats.num_xor}, NOT: ${stats.num_not}, MUX: ${stats.num_mux})<br>
                    <strong>Critical Path Delay:</strong> ${stats.critical_path_delay.toFixed(2)} STA topological delay units<br>
                    <strong>Mean Toggle Activity:</strong> ${stats.mean_toggle_rate.toFixed(4)} transitions / cycle<br>
                    <strong>Primary IOs:</strong> ${stats.num_primary_inputs} Inputs / ${stats.num_primary_outputs} Outputs
                `;

            } catch (err) {
                log(`[ERROR] Network or server error: ${err}`, 'error');
            } finally {
                btn.disabled = false;
                btn.innerText = '▶ Run Presynthesis PPA';
            }
        }

        window.onload = init;
    </script>
</body>
</html>
"""


class MasterRTLEDAHandler(http.server.SimpleHTTPRequestHandler):
    """Custom HTTP Request Handler serving EDA Studio and PPA API."""

    def log_message(self, format: str, *args: Any) -> None:
        pass  # Suppress default noisy console logs

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path in {"/", "/index.html", "/gui"}:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(STUDIO_HTML.encode("utf-8"))
            return

        if path == "/api/designs":
            raw_dir = PROJECT_ROOT / "data" / "raw_rtl"
            files = sorted([f.name for f in raw_dir.glob("*.v")]) if raw_dir.is_dir() else []
            self._send_json(files)
            return

        if path == "/api/design":
            query = urllib.parse.parse_qs(parsed.query)
            name = query.get("name", [""])[0]
            raw_file = PROJECT_ROOT / "data" / "raw_rtl" / name
            if raw_file.is_file():
                content = raw_file.read_text(encoding="utf-8")
                self._send_json({"name": name, "code": content})
            else:
                self._send_json({"error": "File not found"}, status=404)
            return

        if path.startswith("/reports/"):
            # Serve files from data/reports
            rel_file = path.replace("/reports/", "")
            target = PROJECT_ROOT / "data" / "reports" / rel_file
            if target.is_file():
                self.send_response(200)
                if target.suffix == ".png":
                    self.send_header("Content-Type", "image/png")
                elif target.suffix == ".pdf":
                    self.send_header("Content-Type", "application/pdf")
                elif target.suffix == ".html":
                    self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(target.read_bytes())
                return

        self.send_response(404)
        self.end_headers()

    def do_POST(self) -> None:
        if self.path == "/api/analyze":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            req = json.loads(body.decode("utf-8"))

            design_name = req.get("design_name", "custom_design.v")
            code = req.get("code", "")
            clk = float(req.get("clk_period", 1.0))
            tech = req.get("target_tech") or None
            gen_plot = req.get("generate_plot", True)

            try:
                model = get_model()

                # If custom code or non-existent file, write to a temp directory
                raw_file = PROJECT_ROOT / "data" / "raw_rtl" / design_name
                if not raw_file.is_file() or code.strip() != raw_file.read_text(encoding="utf-8").strip():
                    temp_dir = PROJECT_ROOT / "data" / "inference_temp"
                    temp_dir.mkdir(parents=True, exist_ok=True)
                    target_file = temp_dir / design_name
                    target_file.write_text(code, encoding="utf-8")
                else:
                    target_file = raw_file

                res = predict_verilog_file(
                    verilog_file=target_file,
                    model=model,
                    clk_period=clk,
                    target_tech=tech,
                )

                plot_url = None
                if gen_plot:
                    pred_dir = PROJECT_ROOT / "data" / "reports" / "predictions"
                    png_p, _ = generate_prediction_plot(res, pred_dir, clk_period=clk)
                    plot_url = f"/reports/predictions/{png_p.name}"

                res_copy = dict(res)
                res_copy.pop("paths", None)  # Clean non-serializable graph objects
                res_copy["plot_url"] = plot_url
                self._send_json(res_copy)
            except Exception as e:
                self._send_json({"error": str(e)}, status=500)
            return

        self.send_response(404)
        self.end_headers()

    def _send_json(self, data: Any, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))


class ReusableTCPServer(socketserver.TCPServer):
    allow_reuse_address = True


def start_gui_server(port: int = 8080) -> None:
    """Start the EDA Studio server synchronously."""
    with ReusableTCPServer(("", port), MasterRTLEDAHandler) as httpd:
        print(f"\n[EDA STUDIO] Server online at: http://localhost:{port}")
        print("[EDA STUDIO] Press Ctrl+C to stop.\n")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[EDA STUDIO] Shutting down.")


def start_gui_server_thread(port: int = 8080) -> threading.Thread:
    """Start the EDA Studio server in a daemon background thread."""
    def run_server():
        try:
            with ReusableTCPServer(("", port), MasterRTLEDAHandler) as httpd:
                httpd.serve_forever()
        except Exception:
            pass

    t = threading.Thread(target=run_server, daemon=True)
    t.start()
    time.sleep(0.5)
    return t


def main() -> int:
    parser = argparse.ArgumentParser(description="MasterRTL Interactive EDA Web Studio")
    parser.add_argument("--port", type=int, default=8080, help="Port to run GUI server on (default: 8080)")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically open web browser")
    args = parser.parse_args()

    if not args.no_browser:
        import webbrowser
        threading.Timer(0.8, lambda: webbrowser.open(f"http://localhost:{args.port}")).start()

    start_gui_server(port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
