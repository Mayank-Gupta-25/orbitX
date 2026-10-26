/**
 * OrbitX 3D Space Scene & Rendering Engine
 * Built with Three.js (WebGL)
 * Handles Earth, Atmosphere, Starfield, Lighting, Camera modes & Transitions.
 */

class SpaceScene {
  constructor(containerId) {
    this.container = document.getElementById(containerId);
    this.scale = 1.0 / 1000.0; // 1 unit = 1000 km (Earth R = 6.378 units)
    this.earthRadius = 6.378;

    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(
      45,
      this.container.clientWidth / this.container.clientHeight,
      0.01,
      2000
    );
    this.camera.position.set(0, 12, 22);

    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false, powerPreference: 'high-performance' });
    this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.1;
    this.container.appendChild(this.renderer.domElement);

    this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.05;
    this.controls.minDistance = 6.8;
    this.controls.maxDistance = 250;

    this.cameraMode = 'global'; // 'global' | 'chase' | 'encounter'
    this.targetFocusPos = new THREE.Vector3();
    this.earthRotationSpeed = 0.0005;
    this.enableEarthRotation = true;

    this.initLighting();
    this.initStarfield();
    this.initEarth();
    this.initAtmosphere();

    window.addEventListener('resize', () => this.onWindowResize());
  }

  initLighting() {
    // Ambient cosmic background light
    this.ambientLight = new THREE.AmbientLight(0x1a2639, 0.8);
    this.scene.add(this.ambientLight);

    // Sun directional light (Simulating solar illumination from ECI coordinates)
    this.sunLight = new THREE.DirectionalLight(0xffffff, 2.2);
    this.sunLight.position.set(200, 40, 150).normalize();
    this.scene.add(this.sunLight);

    // Subtle blue rim light from opposite side
    const rimLight = new THREE.DirectionalLight(0x00f0ff, 0.4);
    rimLight.position.set(-200, -20, -150).normalize();
    this.scene.add(rimLight);
  }

  initStarfield() {
    const starCanvas = EarthTextureGenerator.createStarfieldCanvas(2048, 1024);
    const starTexture = new THREE.CanvasTexture(starCanvas);
    starTexture.colorSpace = THREE.SRGBColorSpace;

    const starGeo = new THREE.SphereGeometry(600, 32, 32);
    const starMat = new THREE.MeshBasicMaterial({
      map: starTexture,
      side: THREE.BackSide,
    });
    this.starfield = new THREE.Mesh(starGeo, starMat);
    this.scene.add(this.starfield);
  }

  initEarth() {
    this.earthGroup = new THREE.Group();
    this.scene.add(this.earthGroup);

    // Earth Diffuse Texture
    const earthCanvas = EarthTextureGenerator.createEarthDiffuseCanvas(2048, 1024);
    const earthTexture = new THREE.CanvasTexture(earthCanvas);
    earthTexture.colorSpace = THREE.SRGBColorSpace;

    // Earth Sphere Geometry (Tilted on 23.44° axial obliquity)
    const earthGeo = new THREE.SphereGeometry(this.earthRadius, 64, 64);
    const earthMat = new THREE.MeshStandardMaterial({
      map: earthTexture,
      roughness: 0.75,
      metalness: 0.1,
    });
    this.earthMesh = new THREE.Mesh(earthGeo, earthMat);
    this.earthGroup.add(this.earthMesh);

    // 23.44° Earth Axial Tilt
    this.earthGroup.rotation.z = (23.44 * Math.PI) / 180;

    // Clouds Sphere
    const cloudCanvas = EarthTextureGenerator.createCloudCanvas(2048, 1024);
    const cloudTexture = new THREE.CanvasTexture(cloudCanvas);
    cloudTexture.colorSpace = THREE.SRGBColorSpace;

    const cloudGeo = new THREE.SphereGeometry(this.earthRadius * 1.008, 64, 64);
    const cloudMat = new THREE.MeshStandardMaterial({
      map: cloudTexture,
      transparent: true,
      opacity: 0.85,
      blending: THREE.NormalBlending,
      depthWrite: false,
    });
    this.cloudMesh = new THREE.Mesh(cloudGeo, cloudMat);
    this.earthGroup.add(this.cloudMesh);
  }

  initAtmosphere() {
    // Custom Rayleigh Scattering Atmosphere Limb Glow Shader
    const atmosphereVertex = `
      varying vec3 vNormal;
      varying vec3 vPosition;
      void main() {
        vNormal = normalize(normalMatrix * normal);
        vPosition = (modelViewMatrix * vec4(position, 1.0)).xyz;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }
    `;

    const atmosphereFragment = `
      varying vec3 vNormal;
      varying vec3 vPosition;
      void main() {
        vec3 viewDir = normalize(-vPosition);
        float intensity = pow(1.0 - max(dot(viewDir, vNormal), 0.0), 3.0);
        vec3 atmosphereColor = vec3(0.15, 0.65, 1.0); // Cyan-blue Rayleigh limb
        gl_FragColor = vec4(atmosphereColor, intensity * 0.9);
      }
    `;

    const atmosGeo = new THREE.SphereGeometry(this.earthRadius * 1.025, 48, 48);
    const atmosMat = new THREE.ShaderMaterial({
      vertexShader: atmosphereVertex,
      fragmentShader: atmosphereFragment,
      blending: THREE.AdditiveBlending,
      side: THREE.BackSide,
      transparent: true,
      depthWrite: false,
    });

    this.atmosphereMesh = new THREE.Mesh(atmosGeo, atmosMat);
    this.scene.add(this.atmosphereMesh);
  }

  setCameraMode(mode, targetPos = null) {
    this.cameraMode = mode;
    if (mode === 'global') {
      this.controls.target.set(0, 0, 0);
      this.controls.minDistance = 6.8;
      this.controls.maxDistance = 250;
      this.animateCameraTo(new THREE.Vector3(0, 12, 22), new THREE.Vector3(0, 0, 0));
    } else if (mode === 'chase' && targetPos) {
      this.controls.minDistance = 0.05;
      this.controls.maxDistance = 20;
      const offset = targetPos.clone().normalize().multiplyScalar(0.4);
      offset.y += 0.2;
      this.animateCameraTo(targetPos.clone().add(offset), targetPos);
    } else if (mode === 'encounter' && targetPos) {
      this.controls.minDistance = 0.01;
      this.controls.maxDistance = 15;
      const offset = new THREE.Vector3(0.15, 0.08, 0.2);
      this.animateCameraTo(targetPos.clone().add(offset), targetPos);
    }
  }

  animateCameraTo(destPos, destTarget, duration = 1200) {
    const startPos = this.camera.position.clone();
    const startTarget = this.controls.target.clone();
    const startTime = performance.now();

    const animateTransition = (currentTime) => {
      const elapsed = currentTime - startTime;
      const t = Math.min(elapsed / duration, 1.0);
      // Smooth easeInOutCubic
      const ease = t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;

      this.camera.position.lerpVectors(startPos, destPos, ease);
      this.controls.target.lerpVectors(startTarget, destTarget, ease);
      this.controls.update();

      if (t < 1.0) {
        requestAnimationFrame(animateTransition);
      }
    };
    requestAnimationFrame(animateTransition);
  }

  onWindowResize() {
    this.camera.aspect = this.container.clientWidth / this.container.clientHeight;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
  }

  update(deltaTime) {
    // Earth and Cloud Rotation
    if (this.enableEarthRotation) {
      this.earthMesh.rotation.y += this.earthRotationSpeed;
      this.cloudMesh.rotation.y += this.earthRotationSpeed * 1.15; // Clouds drift faster
    }
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }
}
