/**
 * OrbitX Orbital Mechanics & Object Renderer
 * Renders target satellite, debris catalog, orbit paths, motion trails,
 * dynamic encounter vectors, and RL policy avoidance trajectories.
 */

class OrbitRenderer {
  constructor(spaceScene) {
    this.spaceScene = spaceScene;
    this.scene = spaceScene.scene;
    this.scale = spaceScene.scale; // 1 unit = 1000 km

    this.showOrbits = true;
    this.showTrails = true;
    this.showLabels = true;
    this.showDebrisCloud = true;

    // Groups
    this.orbitGroup = new THREE.Group();
    this.objectGroup = new THREE.Group();
    this.encounterGroup = new THREE.Group();
    this.rlGroup = new THREE.Group();

    this.scene.add(this.orbitGroup);
    this.scene.add(this.objectGroup);
    this.scene.add(this.encounterGroup);
    this.scene.add(this.rlGroup);

    // State cache
    this.targetState = null;
    this.selectedDebrisState = null;
    this.currentTcaData = null;

    this.initTargetMesh();
    this.initDebrisCatalog();
    this.initDynamicEncounterLine();
  }

  initTargetMesh() {
    // Aerospace Satellite Body (Bus + Dual Solar Arrays + High-Gain Dish)
    this.targetGroup = new THREE.Group();

    // Central Bus (Gold MLI Thermal Foil)
    const busGeo = new THREE.BoxGeometry(0.04, 0.04, 0.06);
    const busMat = new THREE.MeshStandardMaterial({
      color: 0xffd700,
      metalness: 0.9,
      roughness: 0.2,
      emissive: 0x332a00,
    });
    const busMesh = new THREE.Mesh(busGeo, busMat);
    this.targetGroup.add(busMesh);

    // Solar Arrays (Silicon Blue Panels)
    const panelGeo = new THREE.BoxGeometry(0.14, 0.005, 0.05);
    const panelMat = new THREE.MeshStandardMaterial({
      color: 0x1e3a8a,
      metalness: 0.8,
      roughness: 0.3,
      emissive: 0x0f172a,
    });
    const leftPanel = new THREE.Mesh(panelGeo, panelMat);
    leftPanel.position.set(-0.1, 0, 0);
    const rightPanel = new THREE.Mesh(panelGeo, panelMat);
    rightPanel.position.set(0.1, 0, 0);
    this.targetGroup.add(leftPanel);
    this.targetGroup.add(rightPanel);

    // Target Cyan Beacon Glow Ring
    const beaconGeo = new THREE.RingGeometry(0.08, 0.09, 32);
    const beaconMat = new THREE.MeshBasicMaterial({
      color: 0x00f0ff,
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 0.8,
    });
    this.beaconMesh = new THREE.Mesh(beaconGeo, beaconMat);
    this.targetGroup.add(this.beaconMesh);

    this.objectGroup.add(this.targetGroup);

    // Target Motion Trail
    this.targetTrailPoints = [];
    const trailGeo = new THREE.BufferGeometry();
    const trailMat = new THREE.LineBasicMaterial({
      color: 0x00f0ff,
      transparent: true,
      opacity: 0.6,
      blending: THREE.AdditiveBlending,
    });
    this.targetTrailLine = new THREE.Line(trailGeo, trailMat);
    this.objectGroup.add(this.targetTrailLine);
  }

  initDebrisCatalog() {
    // High-performance Point Cloud for Background Debris Catalog (1000+ objects)
    const count = 1200;
    const positions = new Float32Array(count * 3);
    const colors = new Float32Array(count * 3);

    // Distribute in realistic LEO orbital altitude bands (500 km - 1000 km)
    for (let i = 0; i < count; i++) {
      const altKm = 6378 + 450 + Math.random() * 550;
      const r = altKm * this.scale;
      const theta = Math.random() * Math.PI * 2;
      const phi = (Math.random() - 0.5) * Math.PI * 0.85; // Inclinations up to 75 deg

      positions[i * 3] = r * Math.cos(phi) * Math.cos(theta);
      positions[i * 3 + 1] = r * Math.sin(phi);
      positions[i * 3 + 2] = r * Math.cos(phi) * Math.sin(theta);

      // Muted slate gray with subtle red-orange hazard debris tint
      colors[i * 3] = 0.8;
      colors[i * 3 + 1] = 0.3 + Math.random() * 0.3;
      colors[i * 3 + 2] = 0.2;
    }

    const debrisGeo = new THREE.BufferGeometry();
    debrisGeo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    debrisGeo.setAttribute('color', new THREE.BufferAttribute(colors, 3));

    const debrisMat = new THREE.PointsMaterial({
      size: 0.05,
      vertexColors: true,
      transparent: true,
      opacity: 0.7,
      blending: THREE.AdditiveBlending,
    });

    this.debrisPointCloud = new THREE.Points(debrisGeo, debrisMat);
    this.objectGroup.add(this.debrisPointCloud);

    // Dedicated Selected Debris High-Priority Marker
    this.selectedDebrisMarker = new THREE.Group();
    const debCore = new THREE.Mesh(
      new THREE.DodecahedronGeometry(0.04),
      new THREE.MeshStandardMaterial({ color: 0xef4444, emissive: 0xff0000, metalness: 0.9 })
    );
    const debRing = new THREE.Mesh(
      new THREE.RingGeometry(0.07, 0.08, 32),
      new THREE.MeshBasicMaterial({ color: 0xef4444, side: THREE.DoubleSide })
    );
    this.selectedDebrisMarker.add(debCore);
    this.selectedDebrisMarker.add(debRing);
    this.selectedDebrisMarker.visible = false;
    this.objectGroup.add(this.selectedDebrisMarker);
  }

  initDynamicEncounterLine() {
    // Dynamic Red/Amber Separation Line between Target and Selected Debris
    const lineGeo = new THREE.BufferGeometry();
    lineGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(6), 3));
    const lineMat = new THREE.LineDashedMaterial({
      color: 0xef4444,
      dashSize: 0.05,
      gapSize: 0.03,
      transparent: true,
      opacity: 0.9,
    });
    this.separationLine = new THREE.Line(lineGeo, lineMat);
    this.separationLine.computeLineDistances();
    this.separationLine.visible = false;
    this.encounterGroup.add(this.separationLine);

    // Predicted Closest Approach (TCA) Marker
    const tcaGeo = new THREE.SphereGeometry(0.03, 16, 16);
    const tcaMat = new THREE.MeshBasicMaterial({ color: 0xf59e0b, wireframe: true });
    this.tcaMarker = new THREE.Mesh(tcaGeo, tcaMat);
    this.tcaMarker.visible = false;
    this.encounterGroup.add(this.tcaMarker);
  }

  renderTargetOrbit(orbitPointsKm) {
    // Clear old orbit
    while (this.orbitGroup.children.length > 0) {
      this.orbitGroup.remove(this.orbitGroup.children[0]);
    }

    const points = orbitPointsKm.map(
      (p) => new THREE.Vector3(p[0] * this.scale, p[1] * this.scale, p[2] * this.scale)
    );
    const curve = new THREE.CatmullRomCurve3(points, true);
    const curvePoints = curve.getPoints(360);

    const geo = new THREE.BufferGeometry().setFromPoints(curvePoints);
    const mat = new THREE.LineBasicMaterial({
      color: 0x00f0ff,
      transparent: true,
      opacity: 0.5,
    });
    this.targetOrbitLine = new THREE.Line(geo, mat);
    this.orbitGroup.add(this.targetOrbitLine);
  }

  renderDebrisTrajectory(debrisPointsKm) {
    // Debris trajectory spline
    const points = debrisPointsKm.map(
      (p) => new THREE.Vector3(p[0] * this.scale, p[1] * this.scale, p[2] * this.scale)
    );
    const curve = new THREE.CatmullRomCurve3(points);
    const curvePoints = curve.getPoints(200);

    const geo = new THREE.BufferGeometry().setFromPoints(curvePoints);
    const mat = new THREE.LineBasicMaterial({
      color: 0xef4444,
      transparent: true,
      opacity: 0.7,
    });
    if (this.debrisTrajLine) {
      this.encounterGroup.remove(this.debrisTrajLine);
    }
    this.debrisTrajLine = new THREE.Line(geo, mat);
    this.encounterGroup.add(this.debrisTrajLine);
  }

  setRLAvoidanceTrajectory(nominalPointsKm, avoidancePointsKm, burnPointKm, deltaV) {
    // Clear previous RL trajectories
    while (this.rlGroup.children.length > 0) {
      this.rlGroup.remove(this.rlGroup.children[0]);
    }

    // 1. Nominal (Uncontrolled) Path in RED dashed
    const nomPoints = nominalPointsKm.map(
      (p) => new THREE.Vector3(p[0] * this.scale, p[1] * this.scale, p[2] * this.scale)
    );
    const nomGeo = new THREE.BufferGeometry().setFromPoints(nomPoints);
    const nomMat = new THREE.LineDashedMaterial({
      color: 0xef4444,
      dashSize: 0.08,
      gapSize: 0.04,
      transparent: true,
      opacity: 0.8,
    });
    const nomLine = new THREE.Line(nomGeo, nomMat);
    nomLine.computeLineDistances();
    this.rlGroup.add(nomLine);

    // 2. Policy-Controlled Avoidance Path in GLOWING CYAN
    const avoidPoints = avoidancePointsKm.map(
      (p) => new THREE.Vector3(p[0] * this.scale, p[1] * this.scale, p[2] * this.scale)
    );
    const avoidGeo = new THREE.BufferGeometry().setFromPoints(avoidPoints);
    const avoidMat = new THREE.LineBasicMaterial({
      color: 0x10b981, // Emerald Green Safety Clearance
      linewidth: 2,
    });
    const avoidLine = new THREE.Line(avoidGeo, avoidMat);
    this.rlGroup.add(avoidLine);

    // 3. Maneuver Burn Impulse Arrow
    if (burnPointKm) {
      const burnPos = new THREE.Vector3(
        burnPointKm[0] * this.scale,
        burnPointKm[1] * this.scale,
        burnPointKm[2] * this.scale
      );
      const burnGeo = new THREE.SphereGeometry(0.04, 16, 16);
      const burnMat = new THREE.MeshBasicMaterial({ color: 0x00f0ff });
      const burnMarker = new THREE.Mesh(burnGeo, burnMat);
      burnMarker.position.copy(burnPos);
      this.rlGroup.add(burnMarker);
    }
  }

  updatePositions(targetKm, debrisKm, isEncounterActive = false) {
    // Target Satellite Position
    if (targetKm) {
      const targetVec = new THREE.Vector3(
        targetKm[0] * this.scale,
        targetKm[1] * this.scale,
        targetKm[2] * this.scale
      );
      this.targetGroup.position.copy(targetVec);

      // Face towards orbital motion direction
      if (this.targetTrailPoints.length > 0) {
        const lastPt = this.targetTrailPoints[this.targetTrailPoints.length - 1];
        const dir = targetVec.clone().sub(lastPt).normalize();
        if (dir.lengthSq() > 0.0001) {
          this.targetGroup.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), dir);
        }
      }

      // Update motion trail
      if (this.showTrails) {
        this.targetTrailPoints.push(targetVec.clone());
        if (this.targetTrailPoints.length > 80) {
          this.targetTrailPoints.shift();
        }
        this.targetTrailLine.geometry.setFromPoints(this.targetTrailPoints);
        this.targetTrailLine.visible = true;
      } else {
        this.targetTrailLine.visible = false;
      }

      // Camera auto-follow if in chase mode
      if (this.spaceScene.cameraMode === 'chase') {
        const offset = targetVec.clone().normalize().multiplyScalar(0.4);
        offset.y += 0.15;
        this.spaceScene.camera.position.copy(targetVec.clone().add(offset));
        this.spaceScene.controls.target.copy(targetVec);
      }
    }

    // Selected Debris Position
    if (debrisKm && isEncounterActive) {
      const debrisVec = new THREE.Vector3(
        debrisKm[0] * this.scale,
        debrisKm[1] * this.scale,
        debrisKm[2] * this.scale
      );
      this.selectedDebrisMarker.position.copy(debrisVec);
      this.selectedDebrisMarker.visible = true;

      // Update Dynamic Separation Line
      if (targetKm) {
        const positions = this.separationLine.geometry.attributes.position.array;
        positions[0] = targetKm[0] * this.scale;
        positions[1] = targetKm[1] * this.scale;
        positions[2] = targetKm[2] * this.scale;
        positions[3] = debrisKm[0] * this.scale;
        positions[4] = debrisKm[1] * this.scale;
        positions[5] = debrisKm[2] * this.scale;
        this.separationLine.geometry.attributes.position.needsUpdate = true;
        this.separationLine.computeLineDistances();
        this.separationLine.visible = true;
      }

      // Camera auto-follow if in encounter mode
      if (this.spaceScene.cameraMode === 'encounter' && targetKm) {
        const midpoint = targetKm.map((v, i) => (v + debrisKm[i]) * 0.5 * this.scale);
        const midVec = new THREE.Vector3(midpoint[0], midpoint[1], midpoint[2]);
        this.spaceScene.controls.target.copy(midVec);
      }
    } else {
      this.selectedDebrisMarker.visible = false;
      this.separationLine.visible = false;
    }

    // Toggle visibilities
    this.targetOrbitLine.visible = this.showOrbits;
    this.debrisPointCloud.visible = this.showDebrisCloud;
  }
}
