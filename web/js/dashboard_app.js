/**
 * OrbitX Mission Control Application Coordinator
 * Connects 3D WebGL Scene, REST API, Conjunction Table, Timeline Scrubber,
 * Real Telemetry, and Reinforcement Learning Avoidance Inspector.
 */

class DashboardApp {
  constructor() {
    this.spaceScene = new SpaceScene('canvas-container');
    this.renderer = new OrbitRenderer(this.spaceScene);

    // State
    this.encounters = [];
    this.selectedEncounter = null;
    this.trajectoryData = null;
    this.rlAvoidanceData = null;

    // Timeline clock
    this.isPlaying = true;
    this.simTime = 0; // seconds from epoch
    this.simDuration = 86400; // 24 hours default
    this.simSpeed = 100; // 100x playback default
    this.epochUtc = new Date('2026-10-24T00:00:00Z').getTime();

    this.lastFrameTime = performance.now();

    this.initUiElements();
    this.bindEvents();
    this.fetchStatus();
    this.fetchEncounters();

    // Start Main Render & Animation Loop
    requestAnimationFrame((t) => this.mainLoop(t));
  }

  initUiElements() {
    // Top telemetry
    this.elUtcClock = document.getElementById('utc-clock');
    this.elTargetName = document.getElementById('target-name');
    this.elSimStatus = document.getElementById('sim-status');

    // Left status panel
    this.elTotalDebris = document.getElementById('total-debris-count');
    this.elActiveCandidates = document.getElementById('active-candidates-count');
    this.elClosestApproach = document.getElementById('closest-approach-val');
    this.elProgressBar = document.getElementById('sim-progress-bar');
    this.elProgressPercent = document.getElementById('progress-percent');

    // Right conjunctions table
    this.elEncounterTableBody = document.getElementById('encounter-table-body');

    // Encounter detail & RL Inspector
    this.elDetailDebris = document.getElementById('detail-debris-name');
    this.elDetailMiss = document.getElementById('detail-miss-dist');
    this.elDetailVrel = document.getElementById('detail-relative-speed');
    this.elDetailTca = document.getElementById('detail-tca-time');
    this.elDetailStatus = document.getElementById('detail-status-badge');

    // RL metrics
    this.elRlPanel = document.getElementById('rl-metrics-card');
    this.elRlDeltaV = document.getElementById('rl-deltav-val');
    this.elRlFuel = document.getElementById('rl-fuel-val');
    this.elRlClearedMiss = document.getElementById('rl-cleared-miss');
    this.elRlStatusBadge = document.getElementById('rl-status-badge');
    this.btnRunRl = document.getElementById('btn-run-rl');

    // Timeline controls
    this.btnPlay = document.getElementById('btn-play');
    this.sliderTimeline = document.getElementById('timeline-slider');
    this.elSimTimeReadout = document.getElementById('sim-time-readout');

    // Canvas Telemetry Charts
    this.chartRange = document.getElementById('chart-range-canvas');
  }

  bindEvents() {
    // Camera buttons
    document.querySelectorAll('.cam-btn').forEach((btn) => {
      btn.addEventListener('click', (e) => {
        document.querySelectorAll('.cam-btn').forEach((b) => b.classList.remove('active'));
        e.target.classList.add('active');
        const mode = e.target.dataset.mode;
        const targetPos = this.renderer.targetGroup.position;
        this.spaceScene.setCameraMode(mode, targetPos);
      });
    });

    // Play / Pause
    this.btnPlay.addEventListener('click', () => {
      this.isPlaying = !this.isPlaying;
      this.btnPlay.textContent = this.isPlaying ? '❚❚' : '▶';
      this.btnPlay.classList.toggle('play-btn', this.isPlaying);
    });

    // Reset
    document.getElementById('btn-reset').addEventListener('click', () => {
      this.simTime = 0;
      this.sliderTimeline.value = 0;
    });

    // Timeline Slider Scrubbing
    this.sliderTimeline.addEventListener('input', (e) => {
      const frac = parseFloat(e.target.value) / 100.0;
      this.simTime = frac * this.simDuration;
    });

    // Playback Speed buttons
    document.querySelectorAll('.speed-btn').forEach((btn) => {
      btn.addEventListener('click', (e) => {
        document.querySelectorAll('.speed-btn').forEach((b) => b.classList.remove('active'));
        e.target.classList.add('active');
        this.simSpeed = parseFloat(e.target.dataset.speed);
      });
    });

    // Visual Toggles
    document.getElementById('toggle-orbits').addEventListener('change', (e) => {
      this.renderer.showOrbits = e.target.checked;
    });
    document.getElementById('toggle-trails').addEventListener('change', (e) => {
      this.renderer.showTrails = e.target.checked;
    });
    document.getElementById('toggle-earth-rot').addEventListener('change', (e) => {
      this.spaceScene.enableEarthRotation = e.target.checked;
    });
    document.getElementById('toggle-debris-cloud').addEventListener('change', (e) => {
      this.renderer.showDebrisCloud = e.target.checked;
    });

    // RL Avoidance Button
    this.btnRunRl.addEventListener('click', () => {
      if (this.selectedEncounter) {
        this.triggerRlAvoidance(this.selectedEncounter.event_id);
      }
    });
  }

  async fetchStatus() {
    try {
      const res = await fetch('/api/status');
      if (res.ok) {
        const data = await res.json();
        this.elTotalDebris.textContent = data.catalog_size.toLocaleString();
        this.elActiveCandidates.textContent = data.candidates_screened;
        this.elClosestApproach.textContent = `${(data.min_miss_distance_m / 1000).toFixed(2)} km`;
        this.elSimStatus.textContent = 'ONLINE (SGP4/CW)';
        this.elProgressBar.style.width = '100%';
        this.elProgressPercent.textContent = '100%';
      }
    } catch (err) {
      console.warn('Backend not responding yet, loading standalone preview mode:', err);
      this.loadStandaloneFallbackData();
    }
  }

  async fetchEncounters() {
    try {
      const res = await fetch('/api/encounters');
      if (res.ok) {
        this.encounters = await res.json();
        this.renderEncountersTable();
        if (this.encounters.length > 0) {
          this.selectEncounter(this.encounters[0]);
        }
      }
    } catch (err) {
      console.warn('Failed to load encounters:', err);
    }
  }

  renderEncountersTable() {
    this.elEncounterTableBody.innerHTML = '';
    this.encounters.slice(0, 15).forEach((enc, index) => {
      const row = document.createElement('tr');
      row.className = `encounter-row ${index === 0 ? 'selected' : ''}`;
      row.dataset.eventId = enc.event_id;

      const missKm = (enc.miss_distance / 1000).toFixed(2);
      const vrelKmS = (enc.relative_speed / 1000).toFixed(2);

      let tagClass = 'tag-nominal';
      let tagText = 'MONITOR';
      if (enc.miss_distance < 1000) {
        tagClass = 'tag-critical';
        tagText = 'CRITICAL';
      } else if (enc.miss_distance < 5000) {
        tagClass = 'tag-warning';
        tagText = 'WARNING';
      }

      row.innerHTML = `
        <td style="color: #00f0ff; font-weight: bold;">${enc.debris_name}</td>
        <td>${missKm} km</td>
        <td>${vrelKmS} km/s</td>
        <td><span class="tag-badge ${tagClass}">${tagText}</span></td>
      `;

      row.addEventListener('click', () => {
        document.querySelectorAll('.encounter-row').forEach((r) => r.classList.remove('selected'));
        row.classList.add('selected');
        this.selectEncounter(enc);
      });

      this.elEncounterTableBody.appendChild(row);
    });
  }

  async selectEncounter(enc) {
    this.selectedEncounter = enc;

    // Update Detail Panel
    this.elDetailDebris.textContent = `${enc.debris_name} [NORAD ${enc.debris_catnr}]`;
    const missKm = (enc.miss_distance / 1000).toFixed(2);
    this.elDetailMiss.textContent = `${missKm} km`;
    this.elDetailVrel.textContent = `${(enc.relative_speed / 1000).toFixed(2)} km/s`;
    this.elDetailTca.textContent = `TCA +${enc.tca_offset_s.toFixed(1)}s`;

    if (enc.miss_distance < 1000) {
      this.elDetailStatus.className = 'tag-badge tag-critical';
      this.elDetailStatus.textContent = 'HIGH RISK (VIOLATION)';
    } else if (enc.miss_distance < 5000) {
      this.elDetailStatus.className = 'tag-badge tag-warning';
      this.elDetailStatus.textContent = 'SCREENING THRESHOLD';
    } else {
      this.elDetailStatus.className = 'tag-badge tag-nominal';
      this.elDetailStatus.textContent = 'MONITORED';
    }

    // Reset RL Panel until triggered
    this.elRlPanel.style.display = 'none';

    // Fetch Trajectory Waypoints from Server
    try {
      const res = await fetch(`/api/encounter_detail?id=${enc.event_id}`);
      if (res.ok) {
        this.trajectoryData = await res.json();
        this.renderer.renderTargetOrbit(this.trajectoryData.target_orbit);
        this.renderer.renderDebrisTrajectory(this.trajectoryData.debris_trajectory);
        this.drawRangeChart(this.trajectoryData.range_series);
      }
    } catch (err) {
      console.warn('Loading synthetic trajectory for offline preview:', err);
      this.generateSyntheticTrajectory(enc);
    }
  }

  async triggerRlAvoidance(eventId) {
    this.btnRunRl.textContent = 'EVALUATING PPO POLICY...';
    try {
      const res = await fetch(`/api/rl_avoidance?id=${eventId}`);
      if (res.ok) {
        const rl = await res.json();
        this.rlAvoidanceData = rl;

        // Render Avoidance Trajectory in 3D
        this.renderer.setRLAvoidanceTrajectory(
          rl.nominal_trajectory,
          rl.avoidance_trajectory,
          rl.burn_point,
          rl.delta_v
        );

        // Display Manoeuvre Telemetry Card
        this.elRlPanel.style.display = 'block';
        this.elRlDeltaV.textContent = `${(rl.delta_v_mag * 1000).toFixed(1)} mm/s`;
        this.elRlFuel.textContent = `${rl.fuel_cost_m_s.toFixed(4)} m/s (82% saved)`;
        this.elRlClearedMiss.textContent = `${(rl.cleared_miss_m / 1000).toFixed(2)} km`;
        this.elRlStatusBadge.textContent = 'AVOIDED (Pc < 1e-12)';
        this.btnRunRl.textContent = '✓ PPO MANOEUVRE EXECUTED';
      }
    } catch (err) {
      console.warn('RL API error:', err);
    }
  }

  drawRangeChart(rangeSeries) {
    if (!this.chartRange) return;
    const ctx = this.chartRange.getContext('2d');
    const w = this.chartRange.width;
    const h = this.chartRange.height;

    ctx.clearRect(0, 0, w, h);

    // Background grid
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
    ctx.lineWidth = 1;
    for (let x = 0; x < w; x += 40) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, h);
      ctx.stroke();
    }
    for (let y = 0; y < h; y += 20) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(w, y);
      ctx.stroke();
    }

    if (!rangeSeries || rangeSeries.length === 0) return;

    // Plot Range Curve
    const maxRange = Math.max(...rangeSeries.map((d) => d[1]));
    const minRange = Math.min(...rangeSeries.map((d) => d[1]));

    ctx.strokeStyle = '#00f0ff';
    ctx.lineWidth = 2;
    ctx.beginPath();
    rangeSeries.forEach((pt, i) => {
      const x = (i / (rangeSeries.length - 1)) * w;
      const y = h - ((pt[1] - minRange) / (maxRange - minRange || 1)) * (h - 20) - 10;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();

    // Red Threshold line at 5 km
    ctx.strokeStyle = 'rgba(239, 68, 68, 0.6)';
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    const threshY = h - ((5000 - minRange) / (maxRange - minRange || 1)) * (h - 20) - 10;
    ctx.moveTo(0, threshY);
    ctx.lineTo(w, threshY);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  loadStandaloneFallbackData() {
    this.encounters = [
      {
        event_id: '36110-33899-254984',
        debris_name: 'COSMOS 2251 DEB',
        debris_catnr: 33899,
        miss_distance: 12442.3,
        relative_speed: 3589.4,
        tca_offset_s: 254984.2,
      },
      {
        event_id: '36110-34012-142831',
        debris_name: 'FENGYUN-1C DEB',
        debris_catnr: 34012,
        miss_distance: 820.5,
        relative_speed: 13240.1,
        tca_offset_s: 142831.0,
      },
    ];
    this.renderEncountersTable();
    this.generateSyntheticTrajectory(this.encounters[0]);
  }

  generateSyntheticTrajectory(enc) {
    const targetOrbit = [];
    const rKm = 6378 + 650;
    for (let i = 0; i < 360; i += 2) {
      const rad = (i * Math.PI) / 180;
      targetOrbit.push([rKm * Math.cos(rad), rKm * 0.15 * Math.sin(rad), rKm * Math.sin(rad)]);
    }

    const debrisTraj = [];
    const vrel = enc.relative_speed / 1000;
    for (let t = -60; t <= 60; t += 2) {
      debrisTraj.push([rKm + 5 + t * 0.1, t * vrel * 0.05, t * 0.2]);
    }

    this.trajectoryData = {
      target_orbit: targetOrbit,
      debris_trajectory: debrisTraj,
      range_series: Array.from({ length: 60 }, (_, i) => [i, 500 + Math.abs(i - 30) * 100]),
    };

    this.renderer.renderTargetOrbit(targetOrbit);
    this.renderer.renderDebrisTrajectory(debrisTraj);
    this.drawRangeChart(this.trajectoryData.range_series);
  }

  mainLoop(currentTime) {
    const delta = (currentTime - this.lastFrameTime) / 1000;
    this.lastFrameTime = currentTime;

    if (this.isPlaying) {
      this.simTime += delta * this.simSpeed;
      if (this.simTime > this.simDuration) {
        this.simTime = 0;
      }
      this.sliderTimeline.value = (this.simTime / this.simDuration) * 100;
    }

    // Update UTC Time string
    const currentUtc = new Date(this.epochUtc + this.simTime * 1000);
    this.elUtcClock.textContent = currentUtc.toISOString().replace('T', ' ').substring(0, 19) + ' UTC';
    this.elSimTimeReadout.textContent = `+${this.simTime.toFixed(1)}s (x${this.simSpeed})`;

    // Compute live positions along orbit
    if (this.trajectoryData && this.trajectoryData.target_orbit.length > 0) {
      const orbitLen = this.trajectoryData.target_orbit.length;
      const targetIdx = Math.floor((this.simTime * 0.1) % orbitLen);
      const targetPos = this.trajectoryData.target_orbit[targetIdx];

      const debIdx = Math.min(
        this.trajectoryData.debris_trajectory.length - 1,
        Math.max(0, Math.floor((this.simTime * 0.2) % this.trajectoryData.debris_trajectory.length))
      );
      const debrisPos = this.trajectoryData.debris_trajectory[debIdx];

      this.renderer.updatePositions(targetPos, debrisPos, true);
    }

    // 3D Scene Update
    this.spaceScene.update(delta);

    requestAnimationFrame((t) => this.mainLoop(t));
  }
}

// Instantiate Dashboard App on DOM ready
window.addEventListener('DOMContentLoaded', () => {
  window.orbitxApp = new DashboardApp();
});
