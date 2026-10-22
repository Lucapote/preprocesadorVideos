const API_BASE = '/api';

export async function checkHealth() {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) throw new Error('Backend no responde');
  return res.json();
}

export async function uploadVideo(file) {
  const formData = new FormData();
  formData.append('file', file);

  const res = await fetch(`${API_BASE}/upload`, {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || 'Error al subir el video');
  }

  return res.json();
}

export async function triggerPhase0({ filePath = null, sampleMode = false } = {}) {
  const res = await fetch(`${API_BASE}/pipeline/run-phase-0`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      file_path: filePath,
      sample_mode: sampleMode,
    }),
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || 'Error al iniciar la Fase 0');
  }

  return res.json();
}

export async function detectMarkers({ filePath = null, sampleMode = false } = {}) {
  const res = await fetch(`${API_BASE}/pipeline/detect-markers`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      file_path: filePath,
      sample_mode: sampleMode,
    }),
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || 'Error al ejecutar la Fase 0');
  }

  return res.json();
}

export async function runPhase1({ filePath = null, markers = null, sampleMode = false } = {}) {
  const res = await fetch(`${API_BASE}/pipeline/run-phase-1`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      file_path: filePath,
      markers: markers,
      sample_mode: sampleMode,
    }),
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || 'Error al ejecutar la Fase 1');
  }

  return res.json();
}

export async function getTaskStatus(taskId) {
  const res = await fetch(`${API_BASE}/pipeline/status/${taskId}`);
  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || 'Error al consultar la tarea');
  }
  return res.json();
}

export async function runPhase2({ filePath = null, preRollMs = 180, postRollMs = 300, sampleMode = false } = {}) {
  const res = await fetch(`${API_BASE}/pipeline/run-phase-2`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      file_path: filePath,
      pre_roll_padding_ms: preRollMs,
      post_roll_padding_ms: postRollMs,
      sample_mode: sampleMode,
    }),
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || 'Error al ejecutar la Fase 2');
  }

  return res.json();
}

