#!/usr/bin/env sh
set -eu

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CONFIG_PATH="$ROOT/.scale/product-smoke.json"
LOG_DIR="$ROOT/.agent/logs"
LOG_PATH="$LOG_DIR/product-smoke.json"

mkdir -p "$LOG_DIR"

node - "$CONFIG_PATH" "$LOG_PATH" <<'NODE'
const fs = require('fs');
const cp = require('child_process');
const path = require('path');

const configPath = process.argv[2];
const logPath = process.argv[3];

function writeReport(report) {
  fs.mkdirSync(path.dirname(logPath), { recursive: true });
  fs.writeFileSync(logPath, JSON.stringify(report, null, 2) + '\n', 'utf8');
  process.stdout.write(JSON.stringify(report, null, 2) + '\n');
}

if (!fs.existsSync(configPath)) {
  writeReport({
    version: 1,
    status: 'failed',
    verifiedAt: new Date().toISOString(),
    message: 'Missing .scale/product-smoke.json',
    results: []
  });
  process.exit(1);
}

const config = JSON.parse(fs.readFileSync(configPath, 'utf8').replace(/^\uFEFF/, ''));
const probes = Array.isArray(config.probes) ? config.probes.filter(probe => probe && probe.enabled === true) : [];

if (probes.length === 0) {
  const status = config.emptyProbeBehavior === 'block' ? 'failed' : 'skipped';
  writeReport({
    version: 1,
    status,
    verifiedAt: new Date().toISOString(),
    message: 'No enabled product smoke probes. Enable probes in .scale/product-smoke.json after defining the real product path.',
    results: []
  });
  process.exit(status === 'failed' ? 1 : 0);
}

const results = probes.map((probe) => {
  const startedAt = new Date().toISOString();
  const expectedExitCode = Number.isInteger(probe.expected && probe.expected.exitCode) ? probe.expected.exitCode : 0;
  const command = String(probe.command || '');
  if (!command.trim()) {
    return {
      id: String(probe.id || 'unnamed-probe'),
      description: String(probe.description || ''),
      command,
      expectedExitCode,
      exitCode: 1,
      status: 'failed',
      startedAt,
      endedAt: new Date().toISOString(),
      outputTail: 'Probe command is empty'
    };
  }
  const result = cp.spawnSync(command, {
    cwd: process.cwd(),
    shell: true,
    encoding: 'utf8',
    timeout: Number(config.timeoutMs || 180000)
  });
  const output = String(result.stdout || '') + String(result.stderr || '') + String(result.error ? result.error.message : '');
  const exitCode = typeof result.status === 'number' ? result.status : 1;
  return {
    id: String(probe.id || 'unnamed-probe'),
    description: String(probe.description || ''),
    command,
    expectedExitCode,
    exitCode,
    status: exitCode === expectedExitCode ? 'passed' : 'failed',
    startedAt,
    endedAt: new Date().toISOString(),
    outputTail: output.length > 2000 ? output.slice(-2000) : output
  };
});

const failed = results.filter(result => result.status !== 'passed');
writeReport({
  version: 1,
  status: failed.length === 0 ? 'passed' : 'failed',
  verifiedAt: new Date().toISOString(),
  results
});
process.exit(failed.length === 0 ? 0 : 1);

NODE
