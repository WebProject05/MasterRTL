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
    <title>MasterRTL EDA Studio | Professional Presynthesis PPA Workbench</title>
    <style>
        :root {
            --bg-base: #090d16;
            --bg-panel: #111827;
            --bg-card: #1f2937;
            --bg-sub: #374151;
            --border: #374151;
            --border-highlight: #4b5563;
            --accent: #38bdf8;
            --accent-hover: #0ea5e9;
            --accent-green: #10b981;
            --accent-red: #ef4444;
            --accent-amber: #f59e0b;
            --accent-purple: #a855f7;
            --text-main: #f9fafb;
            --text-muted: #9ca3af;
            --text-dim: #6b7280;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; }
        body { background: var(--bg-base); color: var(--text-main); display: flex; flex-direction: column; height: 100vh; overflow: hidden; }

        /* Topbar Navigation */
        .topbar {
            background: #030712;
            border-bottom: 1px solid var(--border);
            padding: 10px 20px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            height: 56px;
        }
        .brand-area { display: flex; align-items: center; gap: 12px; }
        .brand-icon {
            background: linear-gradient(135deg, #0284c7, #38bdf8);
            color: #030712;
            font-weight: 900;
            padding: 5px 9px;
            border-radius: 6px;
            font-size: 0.95rem;
            letter-spacing: 0.8px;
            box-shadow: 0 0 12px rgba(56, 189, 248, 0.35);
        }
        .brand-title { font-size: 1.12rem; font-weight: 700; color: #ffffff; letter-spacing: -0.2px; }
        .brand-subtitle { font-size: 0.72rem; color: var(--text-muted); }
        .pulse-indicator {
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--accent-green);
            box-shadow: 0 0 8px var(--accent-green);
            margin-right: 6px;
            animation: pulse-dot 2s infinite;
        }
        @keyframes pulse-dot {
            0% { transform: scale(0.95); opacity: 0.8; }
            50% { transform: scale(1.2); opacity: 1; }
            100% { transform: scale(0.95); opacity: 0.8; }
        }
        .topbar-actions { display: flex; align-items: center; gap: 10px; }
        .btn-topbar {
            background: var(--bg-card);
            border: 1px solid var(--border);
            color: var(--text-main);
            padding: 6px 12px;
            border-radius: 6px;
            font-size: 0.78rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
            display: flex;
            align-items: center;
            gap: 6px;
        }
        .btn-topbar:hover { background: var(--bg-sub); border-color: var(--accent); color: var(--accent); }
        .tech-pill {
            background: rgba(56, 189, 248, 0.12);
            border: 1px solid rgba(56, 189, 248, 0.35);
            color: var(--accent);
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 700;
            letter-spacing: 0.3px;
        }

        /* Workspace Layout */
        .workspace-grid { display: flex; flex: 1; overflow: hidden; }

        /* Left Control Panel */
        .sidebar {
            width: 340px;
            background: var(--bg-panel);
            border-right: 1px solid var(--border);
            display: flex;
            flex-direction: column;
            overflow-y: auto;
        }
        .sidebar-section { padding: 14px 16px; border-bottom: 1px solid var(--border); }
        .section-header {
            font-size: 0.72rem;
            font-weight: 800;
            text-transform: uppercase;
            color: var(--text-muted);
            letter-spacing: 0.8px;
            margin-bottom: 8px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .search-box {
            position: relative;
            margin-bottom: 8px;
        }
        .search-box input {
            padding-left: 28px;
            font-size: 0.80rem;
            background: #030712;
            border: 1px solid var(--border);
            border-radius: 6px;
            color: var(--text-main);
            width: 100%;
            padding-top: 7px;
            padding-bottom: 7px;
        }
        .search-icon {
            position: absolute;
            left: 9px;
            top: 50%;
            transform: translateY(-50%);
            color: var(--text-dim);
            font-size: 0.8rem;
        }
        .cat-chips { display: flex; gap: 5px; flex-wrap: wrap; margin-bottom: 8px; }
        .cat-chip {
            background: #030712;
            border: 1px solid var(--border);
            color: var(--text-muted);
            font-size: 0.68rem;
            padding: 2px 7px;
            border-radius: 4px;
            cursor: pointer;
            transition: all 0.15s;
        }
        .cat-chip.active, .cat-chip:hover {
            background: var(--accent);
            color: #030712;
            border-color: var(--accent);
            font-weight: 700;
        }

        select, input {
            width: 100%;
            background: #030712;
            border: 1px solid var(--border);
            color: var(--text-main);
            padding: 8px 10px;
            border-radius: 6px;
            font-size: 0.82rem;
            margin-bottom: 8px;
            outline: none;
            transition: border-color 0.2s;
        }
        select:focus, input:focus { border-color: var(--accent); }
        .field-desc { font-size: 0.70rem; color: var(--text-dim); margin-top: -5px; margin-bottom: 8px; }

        .param-row { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }

        .btn-run {
            background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%);
            color: white;
            font-weight: 700;
            font-size: 0.90rem;
            border: none;
            padding: 12px;
            border-radius: 8px;
            cursor: pointer;
            box-shadow: 0 4px 16px rgba(2, 132, 199, 0.4);
            transition: all 0.2s;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            width: 100%;
        }
        .btn-run:hover { filter: brightness(1.15); transform: translateY(-1px); box-shadow: 0 6px 20px rgba(2, 132, 199, 0.5); }
        .btn-run:disabled { opacity: 0.5; cursor: not-allowed; transform: none; box-shadow: none; }

        /* Main Stage */
        .main-stage { flex: 1; display: flex; flex-direction: column; overflow: hidden; background: #030712; }

        /* KPI Scorecard Row */
        .kpi-row {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 12px;
            padding: 12px 18px;
            background: var(--bg-panel);
            border-bottom: 1px solid var(--border);
        }
        .metric-card {
            background: #030712;
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 12px 14px;
            position: relative;
            overflow: hidden;
            transition: all 0.2s;
        }
        .metric-card:hover { border-color: var(--border-highlight); transform: translateY(-1px); }
        .metric-header {
            font-size: 0.72rem;
            font-weight: 700;
            text-transform: uppercase;
            color: var(--text-muted);
            letter-spacing: 0.5px;
            display: flex;
            justify-content: space-between;
        }
        .metric-val {
            font-size: 1.45rem;
            font-weight: 800;
            color: var(--text-main);
            margin: 4px 0;
            font-family: 'Consolas', monospace;
            letter-spacing: -0.5px;
        }
        .metric-desc { font-size: 0.72rem; color: var(--text-dim); }
        .metric-bar-bg { width: 100%; height: 4px; background: #1f2937; border-radius: 2px; margin-top: 6px; overflow: hidden; }
        .metric-bar-fill { height: 100%; background: var(--accent); border-radius: 2px; transition: width 0.4s; }

        /* Tab Navigation */
        .tab-bar {
            background: #030712;
            border-bottom: 1px solid var(--border);
            display: flex;
            padding: 0 14px;
            gap: 4px;
        }
        .tab-item {
            padding: 11px 16px;
            font-size: 0.82rem;
            font-weight: 600;
            color: var(--text-muted);
            cursor: pointer;
            border-bottom: 2px solid transparent;
            transition: all 0.2s;
            display: flex;
            align-items: center;
            gap: 7px;
        }
        .tab-item:hover { color: var(--text-main); }
        .tab-item.active {
            color: var(--accent);
            border-bottom-color: var(--accent);
            background: rgba(56, 189, 248, 0.05);
        }

        /* Views */
        .tab-view { display: none; flex: 1; overflow: auto; padding: 16px 20px; }
        .tab-view.active { display: block; }

        /* Code Editor Container */
        .editor-container {
            display: flex;
            height: calc(100vh - 380px);
            min-height: 280px;
            border: 1px solid var(--border);
            border-radius: 8px;
            overflow: hidden;
            background: #050811;
        }
        .gutter {
            width: 44px;
            background: #030712;
            border-right: 1px solid var(--border);
            color: var(--text-dim);
            font-family: 'Consolas', monospace;
            font-size: 0.82rem;
            line-height: 1.5;
            padding: 12px 6px;
            text-align: right;
            user-select: none;
            overflow: hidden;
        }
        textarea.editor {
            flex: 1;
            background: #050811;
            color: #38bdf8;
            border: none;
            padding: 12px 14px;
            font-family: 'Consolas', 'Courier New', monospace;
            font-size: 0.85rem;
            line-height: 1.5;
            resize: none;
            outline: none;
            white-space: pre;
            overflow: auto;
        }
        .editor-toolbar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
            font-size: 0.76rem;
            color: var(--text-muted);
        }
        .editor-tag { background: var(--bg-card); padding: 3px 8px; border-radius: 4px; border: 1px solid var(--border); }

        /* Visual Dashboard Tab */
        .plot-container { text-align: center; }
        .plot-container img {
            max-width: 95%;
            max-height: 60vh;
            border-radius: 8px;
            border: 1px solid var(--border);
            box-shadow: 0 8px 24px rgba(0,0,0,0.6);
        }

        /* Stats & Details Grid */
        .details-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 14px; }
        .details-card {
            background: var(--bg-panel);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 16px;
        }
        .details-card h4 {
            font-size: 0.88rem;
            color: var(--accent);
            margin-bottom: 12px;
            display: flex;
            align-items: center;
            gap: 6px;
        }
        .stat-line {
            display: flex;
            justify-content: space-between;
            padding: 6px 0;
            border-bottom: 1px solid rgba(255,255,255,0.05);
            font-size: 0.82rem;
        }
        .stat-line:last-child { border-bottom: none; }
        .stat-lbl { color: var(--text-muted); }
        .stat-val { font-weight: 700; color: var(--text-main); font-family: 'Consolas', monospace; }

        /* Optimization Advisor Styles */
        .tab-pill {
            background: #030712;
            color: #94a3b8;
            border: 1px solid var(--border);
            padding: 5px 12px;
            border-radius: 4px;
            font-size: 0.74rem;
            cursor: pointer;
            transition: all 0.2s;
        }
        .tab-pill.active, .tab-pill:hover {
            background: var(--accent);
            color: #030712;
            font-weight: 700;
        }
        .finding-card {
            background: #030712;
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 10px 14px;
            display: flex;
            align-items: flex-start;
            gap: 12px;
            margin-bottom: 8px;
        }
        .finding-badge {
            font-size: 0.68rem;
            font-weight: 800;
            padding: 3px 8px;
            border-radius: 4px;
            white-space: nowrap;
        }
        .badge-CRITICAL { background: rgba(239, 68, 68, 0.2); color: #ef4444; border: 1px solid #ef4444; }
        .badge-WARNING { background: rgba(245, 158, 11, 0.2); color: #f59e0b; border: 1px solid #f59e0b; }
        .badge-OPTIMAL { background: rgba(16, 185, 129, 0.2); color: #10b981; border: 1px solid #10b981; }
        .badge-INFO { background: rgba(56, 189, 248, 0.2); color: #38bdf8; border: 1px solid #38bdf8; }

        .rec-card {
            background: #030712;
            border-left: 4px solid var(--accent);
            border-radius: 4px;
            padding: 10px 14px;
            font-size: 0.83rem;
            margin-bottom: 8px;
        }
        .rec-title { font-weight: 700; color: #f9fafb; margin-bottom: 4px; font-size: 0.88rem; }
        .rec-gain { color: var(--accent-green); font-weight: 600; margin-top: 4px; font-size: 0.78rem; }
        .rec-impact { color: var(--text-dim); font-size: 0.75rem; margin-top: 2px; }

        .wi-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-top: 10px; }
        .wi-card {
            background: #030712;
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 12px;
            text-align: center;
        }
        .wi-title { font-size: 0.70rem; text-transform: uppercase; color: var(--text-muted); font-weight: 700; }
        .wi-val-base { font-size: 0.82rem; color: var(--text-dim); margin: 4px 0; font-family: 'Consolas', monospace; }
        .wi-val-proj { font-size: 1.25rem; font-weight: 800; font-family: 'Consolas', monospace; margin-bottom: 4px; }
        .wi-gain { font-size: 0.72rem; font-weight: 700; padding: 2px 7px; border-radius: 4px; display: inline-block; }

        /* Quality Radar Table */
        .radar-table { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
        .radar-table th { background: #030712; padding: 8px 12px; text-align: left; border-bottom: 1px solid var(--border); color: var(--text-muted); font-weight: 700; }
        .radar-table td { padding: 9px 12px; border-bottom: 1px solid rgba(255,255,255,0.05); font-family: 'Consolas', monospace; }

        /* Bottom Console Drawer */
        .console-drawer {
            height: 140px;
            background: #030712;
            border-top: 1px solid var(--border);
            display: flex;
            flex-direction: column;
            transition: height 0.25s ease;
        }
        .console-header {
            background: #080d1a;
            border-bottom: 1px solid rgba(255,255,255,0.05);
            padding: 5px 16px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 0.72rem;
            font-weight: 700;
            color: var(--text-muted);
        }
        .console-body {
            flex: 1;
            padding: 8px 16px;
            overflow-y: auto;
            font-family: 'Consolas', monospace;
            font-size: 0.78rem;
        }
        .console-line { margin: 2px 0; color: #cbd5e1; }
        .console-line.info { color: #38bdf8; }
        .console-line.success { color: #34d399; }
        .console-line.warn { color: #fbbf24; }
        .console-line.error { color: #f87171; }
    </style>
</head>
<body>

    <!-- Header Navigation -->
    <header class="topbar">
        <div class="brand-area">
            <div class="brand-icon">RTL</div>
            <div>
                <div class="brand-title">MasterRTL EDA Studio <span style="font-size: 0.70rem; color: var(--accent); font-weight: 400; border: 1px solid rgba(56,189,248,0.4); border-radius: 4px; padding: 1px 5px; margin-left: 6px;">v2.4 Pro</span></div>
                <div class="brand-subtitle"><span class="pulse-indicator"></span>Presynthesis PPA Simulation & Optimization Workbench</div>
            </div>
        </div>

        <div style="display: flex; align-items: center; gap: 12px;">
            <div class="tech-pill" id="current-node-badge">NanGate 45nm @ 1.00 GHz (1.000 ns)</div>
            <button class="btn-topbar" onclick="exportReportJSON()" title="Export Machine-Readable JSON Report">📥 Export JSON</button>
            <button class="btn-topbar" onclick="openPlotTab()" title="View and Download 4-Panel Plots">🖼 Plots</button>
        </div>
    </header>

    <!-- Main Workspace -->
    <div class="workspace-grid">
        
        <!-- Left Sidebar Navigator & Controls -->
        <aside class="sidebar">
            <div class="sidebar-section">
                <div class="section-header">
                    <span>1. Benchmark Explorer</span>
                    <span id="benchmark-count" style="color: var(--accent); font-size: 0.70rem;">-- designs</span>
                </div>
                
                <div class="search-box">
                    <span class="search-icon">🔍</span>
                    <input type="text" id="design-search" placeholder="Filter 156+ benchmarks..." oninput="filterDesigns()">
                </div>

                <div class="cat-chips">
                    <span class="cat-chip active" onclick="setCategory(this, '')">All</span>
                    <span class="cat-chip" onclick="setCategory(this, 'adder')">Adders</span>
                    <span class="cat-chip" onclick="setCategory(this, 'multiplier')">Mult</span>
                    <span class="cat-chip" onclick="setCategory(this, 'aes')">Crypto</span>
                    <span class="cat-chip" onclick="setCategory(this, 'fir')">DSP</span>
                    <span class="cat-chip" onclick="setCategory(this, 'riscv')">CPU</span>
                </div>

                <select id="design-select" onchange="loadSelectedDesign()">
                    <option value="">-- Choose Benchmark Design --</option>
                </select>
                <div class="field-desc">Select benchmark or paste custom Verilog RTL into the editor tab.</div>
            </div>

            <div class="sidebar-section">
                <div class="section-header">
                    <span>2. Process Node & Timing</span>
                </div>
                
                <label style="font-size: 0.74rem; color: var(--text-muted); display: block; margin-bottom: 3px;">Foundry Node Scaling</label>
                <select id="tech-select" onchange="updateTechCorner()">
                    <option value="">NanGate 45nm (Open Cell Library - 1.10V)</option>
                    <option value="TSMC_28nm_TYP">TSMC 28nm Typical Corner (0.90V Nominal)</option>
                    <option value="TSMC_28nm_MIN">TSMC 28nm Fast Corner (1.00V Overdrive)</option>
                    <option value="TSMC_28nm_MAX">TSMC 28nm Slow Corner (0.80V Underdrive)</option>
                </select>

                <div class="param-row">
                    <div>
                        <label style="font-size: 0.74rem; color: var(--text-muted); display: block; margin-bottom: 3px;">Clock Period (ns)</label>
                        <input type="number" id="clk-input" value="1.0" step="0.05" min="0.2" oninput="syncClock(this, 'period')">
                    </div>
                    <div>
                        <label style="font-size: 0.74rem; color: var(--text-muted); display: block; margin-bottom: 3px;">Frequency (GHz)</label>
                        <input type="number" id="freq-input" value="1.00" step="0.05" min="0.1" oninput="syncClock(this, 'freq')">
                    </div>
                </div>
            </div>

            <div class="sidebar-section">
                <div class="section-header">
                    <span>3. RTL Templates</span>
                </div>
                <select id="template-select" onchange="loadTemplate(this.value)">
                    <option value="">-- Load Verilog Template --</option>
                    <option value="adder8">8-bit Ripple-Carry Adder</option>
                    <option value="cla16">16-bit Carry-Lookahead Adder</option>
                    <option value="alu32">32-bit Arithmetic Logic Unit (ALU)</option>
                    <option value="shiftreg">8-bit Shift Register with Load</option>
                    <option value="fir">FIR Digital Filter Datapath</option>
                </select>
            </div>

            <div class="sidebar-section" style="flex: 1; display: flex; flex-direction: column; justify-content: flex-end;">
                <button class="btn-run" id="btn-run" onclick="runAnalysis()">
                    <span>▶ Run Presynthesis PPA</span>
                    <span style="font-size: 0.72rem; opacity: 0.8; font-weight: 400;">(Ctrl+Enter)</span>
                </button>
            </div>
        </aside>

        <!-- Main Display Stage -->
        <main class="main-stage">
            
            <!-- Live KPI Banner -->
            <div class="kpi-row">
                <div class="metric-card">
                    <div class="metric-header">
                        <span>Total Silicon Area</span>
                        <span style="color: var(--accent);">NanGate 45nm</span>
                    </div>
                    <div class="metric-val" id="val-area">--</div>
                    <div class="metric-desc" id="sub-area">Seq / Comb Partition</div>
                    <div class="metric-bar-bg"><div class="metric-bar-fill" id="bar-area" style="width: 0%;"></div></div>
                </div>

                <div class="metric-card">
                    <div class="metric-header">
                        <span>Worst Slack (WNS)</span>
                        <span id="badge-wns-state" style="font-size: 0.68rem; font-weight: 700;">--</span>
                    </div>
                    <div class="metric-val" id="val-wns">--</div>
                    <div class="metric-desc" id="sub-wns">Clock Margin Status</div>
                    <div class="metric-bar-bg"><div class="metric-bar-fill" id="bar-wns" style="width: 0%; background: var(--accent-green);"></div></div>
                </div>

                <div class="metric-card">
                    <div class="metric-header">
                        <span>Total Slack (TNS)</span>
                        <span>Setup Violations</span>
                    </div>
                    <div class="metric-val" id="val-tns">--</div>
                    <div class="metric-desc" id="sub-tns">Cumulative Endpoint Slack</div>
                    <div class="metric-bar-bg"><div class="metric-bar-fill" id="bar-tns" style="width: 0%; background: var(--accent-red);"></div></div>
                </div>

                <div class="metric-card">
                    <div class="metric-header">
                        <span>Total Power</span>
                        <span>1.10V @ 1.0 GHz</span>
                    </div>
                    <div class="metric-val" id="val-power">--</div>
                    <div class="metric-desc" id="sub-power">Dynamic Toggle + Leakage</div>
                    <div class="metric-bar-bg"><div class="metric-bar-fill" id="bar-power" style="width: 0%; background: var(--accent-purple);"></div></div>
                </div>
            </div>

            <!-- Tab Bar Navigation -->
            <div class="tab-bar">
                <div class="tab-item active" onclick="switchTab(event, 'view-code')">📄 Verilog RTL Source</div>
                <div class="tab-item" onclick="switchTab(event, 'view-plot')">📈 4-Panel Visual Dashboard</div>
                <div class="tab-item" onclick="switchTab(event, 'view-stats')">🔬 SOG Circuit Structural Topology</div>
                <div class="tab-item" onclick="switchTab(event, 'view-advisor')">💡 PPA Optimization Advisor & What-If</div>
                <div class="tab-item" onclick="switchTab(event, 'view-metrics')">📊 Quality & Efficiency Metrics</div>
            </div>

            <!-- TAB 1: Verilog RTL Source -->
            <div id="view-code" class="tab-view active">
                <div class="editor-toolbar">
                    <div style="display: flex; gap: 8px;">
                        <span class="editor-tag" id="editor-lines">0 lines</span>
                        <span class="editor-tag" id="editor-chars">0 characters</span>
                        <span class="editor-tag" id="editor-design-tag">No design loaded</span>
                    </div>
                    <div>
                        <button class="btn-topbar" style="padding: 3px 8px;" onclick="formatRTL()">Format RTL</button>
                    </div>
                </div>
                <div class="editor-container">
                    <div class="gutter" id="editor-gutter">1</div>
                    <textarea class="editor" id="code-editor" spellcheck="false" oninput="updateGutter()" placeholder="// Select a design from the left or paste synthesizable Verilog code here..."></textarea>
                </div>
            </div>

            <!-- TAB 2: Visual Dashboard -->
            <div id="view-plot" class="tab-view">
                <div class="plot-container">
                    <div id="plot-placeholder" style="color: var(--text-muted); padding: 48px; border: 1px dashed var(--border); border-radius: 8px;">
                        Click <strong>'▶ Run Presynthesis PPA'</strong> to render the publication-grade 4-panel visual dashboard.
                    </div>
                    <div id="plot-view-wrapper" style="display: none;">
                        <div style="display: flex; justify-content: flex-end; gap: 10px; margin-bottom: 12px;">
                            <a id="btn-download-png" class="btn-topbar" href="#" download>📥 Download High-Res PNG (300 DPI)</a>
                            <a id="btn-download-pdf" class="btn-topbar" href="#" download>📄 Download Vector PDF</a>
                        </div>
                        <img id="ppa-plot-img" alt="PPA Visual Dashboard">
                    </div>
                </div>
            </div>

            <!-- TAB 3: SOG Graph Topology -->
            <div id="view-stats" class="tab-view">
                <div class="details-grid">
                    <div class="details-card">
                        <h4>🧩 1-bit Standard Cell Breakdown</h4>
                        <div class="stat-line"><span class="stat-lbl">Total SOG Nodes</span><span class="stat-val" id="sog-total-nodes">--</span></div>
                        <div class="stat-line"><span class="stat-lbl">Sequential Flip-Flops (DFF)</span><span class="stat-val" id="sog-dff-nodes">--</span></div>
                        <div class="stat-line"><span class="stat-lbl">Combinational 1-bit Logic Gates</span><span class="stat-val" id="sog-comb-nodes">--</span></div>
                        <div class="stat-line"><span class="stat-lbl">AND Gates</span><span class="stat-val" id="sog-and">--</span></div>
                        <div class="stat-line"><span class="stat-lbl">OR Gates</span><span class="stat-val" id="sog-or">--</span></div>
                        <div class="stat-line"><span class="stat-lbl">XOR Gates (Arithmetic)</span><span class="stat-val" id="sog-xor">--</span></div>
                        <div class="stat-line"><span class="stat-lbl">Inverters (NOT)</span><span class="stat-val" id="sog-not">--</span></div>
                        <div class="stat-line"><span class="stat-lbl">Multiplexers (MUX)</span><span class="stat-val" id="sog-mux">--</span></div>
                    </div>

                    <div class="details-card">
                        <h4>⚡ Timing & Dynamic Activity Telemetry</h4>
                        <div class="stat-line"><span class="stat-lbl">Critical Path Topological Delay</span><span class="stat-val" id="sog-crit-depth">--</span></div>
                        <div class="stat-line"><span class="stat-lbl">Mean Path Topological Delay</span><span class="stat-val" id="sog-mean-depth">--</span></div>
                        <div class="stat-line"><span class="stat-lbl">Mean Toggle Rate (transitions/cycle)</span><span class="stat-val" id="sog-toggle-rate">--</span></div>
                        <div class="stat-line"><span class="stat-lbl">Weighted Toggle Activity</span><span class="stat-val" id="sog-wt-toggle">--</span></div>
                        <div class="stat-line"><span class="stat-lbl">Primary Input Ports</span><span class="stat-val" id="sog-pi">--</span></div>
                        <div class="stat-line"><span class="stat-lbl">Primary Output Ports</span><span class="stat-val" id="sog-po">--</span></div>
                        <div class="stat-line"><span class="stat-lbl">Evaluated Register-to-Register Paths</span><span class="stat-val" id="sog-reg-paths">--</span></div>
                        <div class="stat-line"><span class="stat-lbl">Standard Cell Library Target</span><span class="stat-val">NanGate 45nm</span></div>
                    </div>
                </div>
            </div>

            <!-- TAB 4: PPA Optimization Advisor & What-If Explorer -->
            <div id="view-advisor" class="tab-view">
                <div style="background: var(--bg-panel); border: 1px solid var(--border); border-radius: 8px; padding: 16px; margin-bottom: 14px;">
                    <h3 style="font-size: 0.95rem; margin-bottom: 10px; color: var(--accent);">1. Circuit Bottlenecks & Diagnostics</h3>
                    <div id="advisor-findings" style="display: flex; flex-direction: column; gap: 8px;">
                        <span style="color: var(--text-muted); font-size: 0.85rem;">Click 'Run Presynthesis PPA' to analyze circuit bottlenecks.</span>
                    </div>
                </div>

                <div style="background: var(--bg-panel); border: 1px solid var(--border); border-radius: 8px; padding: 16px; margin-bottom: 14px;">
                    <h3 style="font-size: 0.95rem; margin-bottom: 10px; color: var(--accent);">2. Actionable RTL Optimization Strategies</h3>
                    <div id="advisor-recs" style="display: flex; flex-direction: column; gap: 8px;">
                        <span style="color: var(--text-muted); font-size: 0.85rem;">Optimization recommendations will be generated based on timing & power profiles.</span>
                    </div>
                </div>

                <div style="background: var(--bg-panel); border: 1px solid var(--border); border-radius: 8px; padding: 16px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <h3 style="font-size: 0.95rem; color: var(--accent);">3. What-If Optimization Exploration</h3>
                        <div style="display: flex; gap: 6px;">
                            <button onclick="setWhatIfScenario('combined')" class="tab-pill active" id="btn-wi-combined">Combined (Pipeline + Gating)</button>
                            <button onclick="setWhatIfScenario('pipelining')" class="tab-pill" id="btn-wi-pipelining">Pipelining Only</button>
                            <button onclick="setWhatIfScenario('clock_gating')" class="tab-pill" id="btn-wi-clock_gating">Clock Gating Only</button>
                        </div>
                    </div>
                    <div id="whatif-container">
                        <span style="color: var(--text-muted); font-size: 0.85rem;">Run analysis to project what-if PPA improvements.</span>
                    </div>
                </div>
            </div>

            <!-- TAB 5: Quality & Efficiency Metrics -->
            <div id="view-metrics" class="tab-view">
                <div class="details-card" style="margin-bottom: 14px;">
                    <h4>📊 Figure-of-Merit & Quality Indicators</h4>
                    <table class="radar-table">
                        <thead>
                            <tr>
                                <th>Figure of Merit (FOM)</th>
                                <th>Computed Value</th>
                                <th>Unit</th>
                                <th>Industrial Interpretation</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td style="color: var(--text-main); font-weight: 600;">Power-Delay Product (PDP)</td>
                                <td id="fom-pdp" style="color: var(--accent);">--</td>
                                <td>fJ / cycle</td>
                                <td>Total switching energy consumed per computation cycle</td>
                            </tr>
                            <tr>
                                <td style="color: var(--text-main); font-weight: 600;">Area-Delay Product (ADP)</td>
                                <td id="fom-adp" style="color: var(--accent);">--</td>
                                <td>μm² · ns</td>
                                <td>Hardware efficiency balancing silicon cost and performance</td>
                            </tr>
                            <tr>
                                <td style="color: var(--text-main); font-weight: 600;">Logic Density</td>
                                <td id="fom-density" style="color: var(--accent);">--</td>
                                <td>gates / μm²</td>
                                <td>Standard cell packaging density under NanGate 45nm</td>
                            </tr>
                            <tr>
                                <td style="color: var(--text-main); font-weight: 600;">Power Efficiency</td>
                                <td id="fom-efficiency" style="color: var(--accent);">--</td>
                                <td>mW / GHz</td>
                                <td>Frequency-normalized dynamic and static power consumption</td>
                            </tr>
                        </tbody>
                    </table>
                </div>

                <div class="details-card">
                    <h4>📐 Model Architecture Reference (Paper Table II)</h4>
                    <div style="font-size: 0.82rem; color: var(--text-muted); line-height: 1.8;">
                        • <strong>Timing Engine:</strong> Random Forest Regressor (80 trees, depth 20) on SOG paths + XGBoost (45 trees, depth 8) calibration.<br>
                        • <strong>Power Engine:</strong> Module-level XGBoost (30 trees, depth 6) evaluating toggle activity + XGBoost (45 trees, depth 8) calibration.<br>
                        • <strong>Area Engine:</strong> Exact analytical Sequential formulation ($N_{\text{DFF}} \times 4.522\text{ μm}^2$) + Combinational XGBoost (45 trees, depth 12).<br>
                        • <strong>Technology Transfer:</strong> Multi-target XGBoost transfer models mapping NanGate 45nm to TSMC 28nm Typical/Min/Max.
                    </div>
                </div>
            </div>

            <!-- Bottom Console Drawer -->
            <div class="console-drawer" id="console-drawer">
                <div class="console-header">
                    <span>TERMINAL CONSOLE & TELEMETRY</span>
                    <div style="display: flex; gap: 8px;">
                        <button class="btn-topbar" style="padding: 2px 7px; font-size: 0.70rem;" onclick="clearConsole()">Clear</button>
                        <button class="btn-topbar" style="padding: 2px 7px; font-size: 0.70rem;" onclick="toggleConsole()" id="btn-toggle-console">▲ Minimize</button>
                    </div>
                </div>
                <div class="console-body" id="console-body">
                    <div class="console-line info">[INIT] MasterRTL EDA Studio v2.4 Pro online. Core ML models loaded.</div>
                    <div class="console-line">[INFO] Ready to elaborate RTL and generate presynthesis PPA estimates.</div>
                </div>
            </div>

        </main>
    </div>

    <script>
        let allDesigns = [];
        let currentResult = null;
        let currentAdvisor = null;

        function log(msg, type = '') {
            const body = document.getElementById('console-body');
            const line = document.createElement('div');
            line.className = 'console-line ' + type;
            const time = new Date().toLocaleTimeString();
            line.innerText = `[${time}] ${msg}`;
            body.appendChild(line);
            body.scrollTop = body.scrollHeight;
        }

        function clearConsole() {
            document.getElementById('console-body').innerHTML = '';
            log('Console cleared.', 'info');
        }

        let consoleMinimized = false;
        function toggleConsole() {
            const drawer = document.getElementById('console-drawer');
            const btn = document.getElementById('btn-toggle-console');
            consoleMinimized = !consoleMinimized;
            if (consoleMinimized) {
                drawer.style.height = '28px';
                btn.innerText = '▼ Expand';
            } else {
                drawer.style.height = '140px';
                btn.innerText = '▲ Minimize';
            }
        }

        function switchTab(evt, tabId) {
            document.querySelectorAll('.tab-view').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-item').forEach(b => b.classList.remove('active'));
            document.getElementById(tabId).classList.add('active');
            evt.currentTarget.classList.add('active');
        }

        function openPlotTab() {
            const tabs = document.querySelectorAll('.tab-item');
            if (tabs.length > 1) {
                tabs[1].click();
            }
        }

        function updateGutter() {
            const text = document.getElementById('code-editor').value;
            const lines = text.split('\\n').length;
            const gutter = document.getElementById('editor-gutter');
            gutter.innerHTML = Array.from({length: lines}, (_, i) => i + 1).join('<br>');
            document.getElementById('editor-lines').innerText = `${lines} lines`;
            document.getElementById('editor-chars').innerText = `${text.length} chars`;
        }

        function syncClock(elem, type) {
            const val = parseFloat(elem.value);
            if (!val || val <= 0) return;
            if (type === 'period') {
                const freq = (1.0 / val).toFixed(2);
                document.getElementById('freq-input').value = freq;
            } else {
                const per = (1.0 / val).toFixed(3);
                document.getElementById('clk-input').value = per;
            }
            updateTechCornerBadge();
        }

        function updateTechCornerBadge() {
            const techSel = document.getElementById('tech-select');
            const techName = techSel.options[techSel.selectedIndex].text.split('(')[0].trim();
            const clk = parseFloat(document.getElementById('clk-input').value) || 1.0;
            const freq = (1.0 / clk).toFixed(2);
            document.getElementById('current-node-badge').innerText = `${techName} @ ${freq} GHz (${clk.toFixed(3)} ns)`;
        }

        function updateTechCorner() {
            updateTechCornerBadge();
            const tech = document.getElementById('tech-select').value;
            log(`Target technology set to: ${tech || 'NanGate 45nm (Baseline)'}`, 'info');
        }

        async function init() {
            try {
                const res = await fetch('/api/designs');
                allDesigns = await res.json();
                document.getElementById('benchmark-count').innerText = `${allDesigns.length} designs`;
                populateDesignSelect(allDesigns);
                log(`Indexed ${allDesigns.length} synthesizable Verilog benchmark designs from data/raw_rtl.`, 'info');

                if (allDesigns.includes('adder_ripple_8bit.v')) {
                    document.getElementById('design-select').value = 'adder_ripple_8bit.v';
                    loadSelectedDesign();
                }
            } catch (e) {
                log(`Failed to list designs: ${e}`, 'error');
            }

            // Keyboard shortcut Ctrl+Enter to run PPA
            document.addEventListener('keydown', (e) => {
                if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
                    e.preventDefault();
                    runAnalysis();
                }
            });
        }

        function populateDesignSelect(list) {
            const sel = document.getElementById('design-select');
            sel.innerHTML = '<option value="">-- Choose Benchmark Design --</option>';
            list.forEach(name => {
                const opt = document.createElement('option');
                opt.value = name;
                opt.innerText = name;
                sel.appendChild(opt);
            });
        }

        function filterDesigns() {
            const q = document.getElementById('design-search').value.toLowerCase().trim();
            const filtered = allDesigns.filter(d => d.toLowerCase().includes(q));
            populateDesignSelect(filtered);
        }

        function setCategory(elem, term) {
            document.querySelectorAll('.cat-chip').forEach(c => c.classList.remove('active'));
            elem.classList.add('active');
            document.getElementById('design-search').value = term;
            filterDesigns();
        }

        async function loadSelectedDesign() {
            const name = document.getElementById('design-select').value;
            if (!name) return;
            log(`Loading RTL source: ${name} ...`, 'info');
            try {
                const res = await fetch(`/api/design?name=${encodeURIComponent(name)}`);
                const data = await res.json();
                document.getElementById('code-editor').value = data.code;
                document.getElementById('editor-design-tag').innerText = name;
                updateGutter();
                log(`Loaded ${name} (${data.code.split('\\n').length} lines).`, 'success');
            } catch (e) {
                log(`Failed to load design: ${e}`, 'error');
            }
        }

        function loadTemplate(tplKey) {
            if (!tplKey) return;
            const templates = {
                adder8: `// 8-bit Ripple-Carry Adder
module adder_ripple_8bit (
    input  wire [7:0] a,
    input  wire [7:0] b,
    input  wire       cin,
    output wire [7:0] sum,
    output wire       cout
);
    wire [8:0] carry;
    assign carry[0] = cin;
    genvar i;
    generate
        for (i = 0; i < 8; i = i + 1) begin : gen_fa
            assign sum[i] = a[i] ^ b[i] ^ carry[i];
            assign carry[i+1] = (a[i] & b[i]) | (carry[i] & (a[i] ^ b[i]));
        end
    endgenerate
    assign cout = carry[8];
endmodule`,
                cla16: `// 16-bit Carry-Lookahead Adder
module cla16 (
    input  wire [15:0] a,
    input  wire [15:0] b,
    input  wire        cin,
    output wire [15:0] sum,
    output wire        cout
);
    wire [15:0] g = a & b;
    wire [15:0] p = a ^ b;
    wire [16:0] c;
    assign c[0] = cin;
    genvar i;
    generate
        for (i = 0; i < 16; i = i + 1) begin : gen_cla
            assign c[i+1] = g[i] | (p[i] & c[i]);
            assign sum[i] = p[i] ^ c[i];
        end
    endgenerate
    assign cout = c[16];
endmodule`,
                alu32: `// 32-bit Arithmetic Logic Unit
module alu32 (
    input  wire [31:0] a,
    input  wire [31:0] b,
    input  wire [2:0]  op,
    output reg  [31:0] result,
    output wire        zero
);
    always @(*) begin
        case (op)
            3'b000: result = a + b;
            3'b001: result = a - b;
            3'b010: result = a & b;
            3'b011: result = a | b;
            3'b100: result = a ^ b;
            3'b101: result = a << b[4:0];
            3'b110: result = a >> b[4:0];
            default: result = 32'b0;
        endcase
    end
    assign zero = (result == 32'b0);
endmodule`,
                shiftreg: `// 8-bit Shift Register with Parallel Load
module shift_reg_8bit (
    input  wire       clk,
    input  wire       rst_n,
    input  wire       load,
    input  wire [7:0] d_in,
    input  wire       shift_in,
    output reg  [7:0] q
);
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            q <= 8'b0;
        else if (load)
            q <= d_in;
        else
            q <= {q[6:0], shift_in};
    end
endmodule`,
                fir: `// 4-Tap FIR Filter Datapath
module fir_filter_4tap (
    input  wire       clk,
    input  wire       rst_n,
    input  wire [7:0] sample_in,
    output reg  [15:0] sample_out
);
    reg [7:0] x0, x1, x2, x3;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            x0 <= 8'b0; x1 <= 8'b0; x2 <= 8'b0; x3 <= 8'b0;
            sample_out <= 16'b0;
        end else begin
            x0 <= sample_in;
            x1 <= x0;
            x2 <= x1;
            x3 <= x2;
            sample_out <= (x0 * 3) + (x1 * 7) + (x2 * 7) + (x3 * 3);
        end
    end
endmodule`
            };

            const code = templates[tplKey];
            if (code) {
                document.getElementById('code-editor').value = code;
                document.getElementById('editor-design-tag').innerText = `template_${tplKey}.v`;
                updateGutter();
                log(`Loaded Verilog template: ${tplKey}`, 'info');
            }
        }

        function formatRTL() {
            const ed = document.getElementById('code-editor');
            ed.value = ed.value.trim() + '\\n';
            updateGutter();
            log('RTL formatted.', 'info');
        }

        async function runAnalysis() {
            const btn = document.getElementById('btn-run');
            btn.disabled = true;
            btn.innerText = '⚙ Synthesizing & Estimating...';
            
            const designName = document.getElementById('design-select').value || document.getElementById('editor-design-tag').innerText || 'custom_design.v';
            const code = document.getElementById('code-editor').value;
            const tech = document.getElementById('tech-select').value;
            const clk = parseFloat(document.getElementById('clk-input').value) || 1.0;

            log(`[SYNTH] Elaborating '${designName}' into Simple Operator Graph (SOG)...`, 'info');

            try {
                const res = await fetch('/api/analyze', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        design_name: designName,
                        code: code,
                        target_tech: tech,
                        clk_period: clk,
                        generate_plot: true
                    })
                });

                const data = await res.json();
                if (data.error) {
                    log(`[ERROR] Synthesis / Estimation failed: ${data.error}`, 'error');
                    return;
                }

                currentResult = data;
                const stats = data.stats;
                const pred = data.prediction;
                const wns = pred.wns;
                const area = pred.area_um2;
                const pwr = pred.power_mW;
                const tns = pred.tns;

                // 1. Update KPI Scorecard
                document.getElementById('val-area').innerText = `${area.toFixed(1)} μm²`;
                const seqArea = stats.num_registers * 4.522;
                const combArea = Math.max(0, area - seqArea);
                document.getElementById('sub-area').innerText = `Seq: ${seqArea.toFixed(1)} μm² | Comb: ${combArea.toFixed(1)} μm²`;
                document.getElementById('bar-area').style.width = `${Math.min(100, (area / 1000) * 100)}%`;

                document.getElementById('val-wns').innerText = `${wns.toFixed(3)} ns`;
                const wnsMet = wns >= 0;
                document.getElementById('val-wns').style.color = wnsMet ? 'var(--accent-green)' : 'var(--accent-red)';
                const badgeWns = document.getElementById('badge-wns-state');
                badgeWns.innerText = wnsMet ? 'MET' : 'VIOLATED';
                badgeWns.style.color = wnsMet ? 'var(--accent-green)' : 'var(--accent-red)';
                document.getElementById('sub-wns').innerText = wnsMet ? `Clock Margin: +${wns.toFixed(3)} ns` : `Violated by ${Math.abs(wns).toFixed(3)} ns`;
                document.getElementById('bar-wns').style.width = `${Math.min(100, Math.max(10, (1 - Math.abs(wns)/clk) * 100))}%`;
                document.getElementById('bar-wns').style.background = wnsMet ? 'var(--accent-green)' : 'var(--accent-red)';

                document.getElementById('val-tns').innerText = `${tns.toFixed(3)} ns`;
                document.getElementById('val-tns').style.color = tns >= 0 ? 'var(--text-main)' : 'var(--accent-red)';
                document.getElementById('sub-tns').innerText = tns >= 0 ? '0 Endpoint Violations' : 'Cumulative Negative Slack';
                document.getElementById('bar-tns').style.width = tns >= 0 ? '0%' : `${Math.min(100, Math.abs(tns) * 20)}%`;

                document.getElementById('val-power').innerText = `${pwr.toFixed(3)} mW`;
                const toggleRatio = Math.min(0.85, Math.max(0.40, stats.mean_toggle_rate * 4.0));
                document.getElementById('sub-power').innerText = `Dynamic: ${(pwr * toggleRatio).toFixed(3)} mW | Leakage: ${(pwr * (1 - toggleRatio)).toFixed(3)} mW`;
                document.getElementById('bar-power').style.width = `${Math.min(100, (pwr / 5) * 100)}%`;

                // 2. Update SOG Topology Tab
                document.getElementById('sog-total-nodes').innerText = stats.total_nodes;
                document.getElementById('sog-dff-nodes').innerText = `${stats.num_registers} (${seqArea.toFixed(1)} μm²)`;
                document.getElementById('sog-comb-nodes').innerText = stats.num_comb_nodes;
                document.getElementById('sog-and').innerText = stats.num_and;
                document.getElementById('sog-or').innerText = stats.num_or;
                document.getElementById('sog-xor').innerText = stats.num_xor;
                document.getElementById('sog-not').innerText = stats.num_not;
                document.getElementById('sog-mux').innerText = stats.num_mux;

                document.getElementById('sog-crit-depth').innerText = `${stats.critical_path_delay.toFixed(2)} logic units`;
                document.getElementById('sog-mean-depth').innerText = `${stats.mean_path_delay.toFixed(2)} logic units`;
                document.getElementById('sog-toggle-rate').innerText = `${stats.mean_toggle_rate.toFixed(4)}`;
                document.getElementById('sog-wt-toggle').innerText = `${stats.weighted_toggle_rate.toFixed(2)}`;
                document.getElementById('sog-pi').innerText = stats.num_primary_inputs;
                document.getElementById('sog-po').innerText = stats.num_primary_outputs;
                document.getElementById('sog-reg-paths').innerText = stats.total_reg_paths || stats.num_registers;

                // 3. Update Quality Figures of Merit
                const pdp = (pwr * clk * 1000).toFixed(2);
                const adp = (area * Math.max(0.1, clk - wns)).toFixed(1);
                const density = (stats.total_nodes / Math.max(1, area)).toFixed(2);
                const eff = (pwr / (1.0 / clk)).toFixed(3);
                document.getElementById('fom-pdp').innerText = `${pdp}`;
                document.getElementById('fom-adp').innerText = `${adp}`;
                document.getElementById('fom-density').innerText = `${density}`;
                document.getElementById('fom-efficiency').innerText = `${eff}`;

                // 4. Update Visual Plot Dashboard
                if (data.plot_url) {
                    const img = document.getElementById('ppa-plot-img');
                    img.src = data.plot_url + '?t=' + new Date().getTime();
                    document.getElementById('plot-view-wrapper').style.display = 'block';
                    document.getElementById('plot-placeholder').style.display = 'none';
                    document.getElementById('btn-download-png').href = data.plot_url;
                    document.getElementById('btn-download-pdf').href = data.plot_url.replace('.png', '.pdf');
                    log(`[PLOT] Rendered 4-panel visual dashboard.`, 'info');
                }

                // 5. Update Optimization Advisor Tab
                if (data.advisor) {
                    currentAdvisor = data.advisor;
                    updateAdvisorUI(data.advisor);
                    log(`[ADVISOR] Diagnosed ${data.advisor.findings.length} findings; generated ${data.advisor.recommendations.length} optimization techniques.`, 'info');
                }

                if (data.transfer) {
                    log(`[TRANSFER] Scaled to ${data.transfer.target_technology}: Area=${data.transfer.transferred_area.toFixed(1)} μm², Power=${data.transfer.transferred_power.toFixed(3)} mW, WNS=${data.transfer.transferred_wns.toFixed(3)} ns`, 'info');
                }

                log(`[SUCCESS] Presynthesis PPA completed in ${data.elapsed_seconds}s. Total Area: ${area.toFixed(1)} μm² | WNS: ${wns.toFixed(3)} ns | Power: ${pwr.toFixed(3)} mW`, 'success');

            } catch (err) {
                log(`[ERROR] Network or server error: ${err}`, 'error');
            } finally {
                btn.disabled = false;
                btn.innerText = '▶ Run Presynthesis PPA';
            }
        }

        function updateAdvisorUI(adv) {
            // 1. Findings
            const fContainer = document.getElementById('advisor-findings');
            fContainer.innerHTML = '';
            (adv.findings || []).forEach(f => {
                const card = document.createElement('div');
                card.className = 'finding-card';
                card.innerHTML = `
                    <span class="finding-badge badge-${f.severity}">${f.severity}</span>
                    <div style="font-size: 0.83rem;">
                        <strong style="color: #f9fafb;">${f.issue}</strong><br>
                        <span style="color: var(--text-muted);">${f.cause}</span>
                    </div>
                `;
                fContainer.appendChild(card);
            });

            // 2. Recommendations
            const rContainer = document.getElementById('advisor-recs');
            rContainer.innerHTML = '';
            (adv.recommendations || []).forEach(r => {
                const card = document.createElement('div');
                card.className = 'rec-card';
                card.innerHTML = `
                    <div class="rec-title">${r.title}</div>
                    <div style="color: #cbd5e1; font-size: 0.82rem;">${r.description}</div>
                    <div class="rec-gain">✦ Expected Gain: ${r.expected_gain}</div>
                    <div class="rec-impact">⚖ Trade-off: ${r.tradeoff}</div>
                `;
                rContainer.appendChild(card);
            });

            // 3. What-If Default
            setWhatIfScenario('combined');
        }

        function setWhatIfScenario(scenario) {
            ['combined', 'pipelining', 'clock_gating'].forEach(s => {
                const btn = document.getElementById('btn-wi-' + s);
                if (btn) {
                    if (s === scenario) btn.classList.add('active');
                    else btn.classList.remove('active');
                }
            });

            const container = document.getElementById('whatif-container');
            if (!currentAdvisor || !currentAdvisor.what_if || !currentAdvisor.what_if[scenario]) {
                container.innerHTML = '<span style="color: var(--text-muted); font-size: 0.85rem;">Scenario data unavailable.</span>';
                return;
            }

            const data = currentAdvisor.what_if[scenario];
            const base = data.baseline;
            const proj = data.projected;
            const imp = data.improvement;

            const wnsClosed = proj.wns >= 0;
            const wnsCol = wnsClosed ? 'var(--accent-green)' : 'var(--accent-red)';
            const wnsBadge = wnsClosed ? 'CLOSURE MET' : 'VIOLATION REDUCED';
            const wnsBadgeCol = wnsClosed ? 'rgba(16, 185, 129, 0.2)' : 'rgba(239, 68, 68, 0.2)';

            container.innerHTML = `
                <div class="wi-grid">
                    <div class="wi-card">
                        <div class="wi-title">Timing Slack (WNS)</div>
                        <div class="wi-val-base">Baseline: ${base.wns.toFixed(3)} ns</div>
                        <div class="wi-val-proj" style="color: ${wnsCol};">${proj.wns.toFixed(3)} ns</div>
                        <div class="wi-gain" style="background: ${wnsBadgeCol}; color: ${wnsCol};">${wnsBadge} (${imp.delta_wns > 0 ? '+' : ''}${imp.delta_wns.toFixed(3)} ns)</div>
                    </div>
                    <div class="wi-card">
                        <div class="wi-title">Total Slack (TNS)</div>
                        <div class="wi-val-base">Baseline: ${base.tns.toFixed(3)} ns</div>
                        <div class="wi-val-proj" style="color: ${proj.tns >= 0 ? 'var(--text-main)' : 'var(--accent-red)'};">${proj.tns.toFixed(3)} ns</div>
                        <div class="wi-gain" style="background: rgba(56, 189, 248, 0.2); color: var(--accent);">${proj.tns >= 0 ? 'All Violations Fixed' : 'Endpoint Slack'}</div>
                    </div>
                    <div class="wi-card">
                        <div class="wi-title">Total Power</div>
                        <div class="wi-val-base">Baseline: ${base.power_mW.toFixed(3)} mW</div>
                        <div class="wi-val-proj" style="color: var(--accent);">${proj.power_mW.toFixed(3)} mW</div>
                        <div class="wi-gain" style="background: rgba(16, 185, 129, 0.2); color: var(--accent-green);">${imp.power_savings_pct}% Power Saved</div>
                    </div>
                    <div class="wi-card">
                        <div class="wi-title">Total Silicon Area</div>
                        <div class="wi-val-base">Baseline: ${base.area_um2.toFixed(1)} μm²</div>
                        <div class="wi-val-proj" style="color: var(--text-main);">${proj.area_um2.toFixed(1)} μm²</div>
                        <div class="wi-gain" style="background: rgba(245, 158, 11, 0.2); color: var(--accent-amber);">${imp.area_overhead_pct > 0 ? '+' : ''}${imp.area_overhead_pct}% Overhead (DFFs)</div>
                    </div>
                </div>
            `;
        }

        function exportReportJSON() {
            if (!currentResult) {
                alert("Please run Presynthesis PPA on a design first.");
                return;
            }
            const blob = new Blob([JSON.stringify(currentResult, null, 2)], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `${currentResult.top_module || 'design'}_master_rtl_report.json`;
            a.click();
            URL.revokeObjectURL(url);
            log(`Exported JSON report for ${currentResult.top_module}.`, 'success');
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
