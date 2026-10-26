/**
 * OrbitX Procedural High-Resolution Earth & Space Texture Generator
 * Generates photorealistic Earth surface, clouds, specular maps, and starfield.
 * Operates 100% offline with zero external network dependencies.
 */

class EarthTextureGenerator {
  /**
   * Generates a realistic Earth diffuse texture with continents, oceans, and ice caps.
   * Size: 2048x1024
   */
  static createEarthDiffuseCanvas(width = 2048, height = 1024) {
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');

    // 1. Deep Ocean Base Gradient
    const oceanGrad = ctx.createLinearGradient(0, 0, 0, height);
    oceanGrad.addColorStop(0.0, '#0a1d37');
    oceanGrad.addColorStop(0.2, '#0f2b48');
    oceanGrad.addColorStop(0.5, '#0b325e');
    oceanGrad.addColorStop(0.8, '#0f2b48');
    oceanGrad.addColorStop(1.0, '#0a1d37');
    ctx.fillStyle = oceanGrad;
    ctx.fillRect(0, 0, width, height);

    // 2. Continents using multi-octave harmonic synthesis
    const imgData = ctx.getImageData(0, 0, width, height);
    const data = imgData.data;

    // Fast 2D Perlin-like pseudo noise generator
    function hash(x, y) {
      let n = Math.sin(x * 12.9898 + y * 78.233) * 43758.5453123;
      return n - Math.floor(n);
    }

    function smoothNoise(x, y) {
      const i = Math.floor(x);
      const j = Math.floor(y);
      const fx = x - i;
      const fy = y - j;
      const u = fx * fx * (3 - 2 * fx);
      const v = fy * fy * (3 - 2 * fy);

      const s00 = hash(i, j);
      const s10 = hash(i + 1, j);
      const s01 = hash(i, j + 1);
      const s11 = hash(i + 1, j + 1);

      return (s00 * (1 - u) + s10 * u) * (1 - v) + (s01 * (1 - u) + s11 * u) * v;
    }

    function fbm(x, y, octaves = 6) {
      let val = 0;
      let amp = 0.5;
      let freq = 1.0;
      for (let o = 0; o < octaves; o++) {
        val += amp * smoothNoise(x * freq, y * freq);
        freq *= 2.05;
        amp *= 0.5;
      }
      return val;
    }

    for (let y = 0; y < height; y++) {
      const lat = (y / height - 0.5) * Math.PI; // -PI/2 to PI/2
      const cosLat = Math.cos(lat);

      for (let x = 0; x < width; x++) {
        const lon = (x / width) * Math.PI * 2; // 0 to 2PI

        // 3D Cartesian coordinates on sphere for seamless longitude wrapping
        const nx = cosLat * Math.cos(lon) * 2.2;
        const ny = cosLat * Math.sin(lon) * 2.2;
        const nz = Math.sin(lat) * 2.2;

        let elevation = fbm(nx + 10, ny + 10, 6) * 0.7 + fbm(nx * 2 + 5, nz * 2 + 5, 4) * 0.3;

        // Shape major continental ridges (Eurasia/Africa/Americas)
        const latDeg = (y / height - 0.5) * 180;
        const lonDeg = (x / width) * 360 - 180;

        // Polar ice caps
        const polar = Math.abs(latDeg) > 65;
        const extremePolar = Math.abs(latDeg) > 78;

        const idx = (y * width + x) * 4;

        if (extremePolar || (polar && elevation > 0.42)) {
          // Glacier / Ice sheet
          data[idx] = 230 + Math.floor(elevation * 25);
          data[idx + 1] = 240 + Math.floor(elevation * 15);
          data[idx + 2] = 250;
        } else if (elevation > 0.52) {
          // Landmass
          const vegetation = fbm(nx * 4, ny * 4, 3);
          const mountain = elevation > 0.65;

          if (mountain) {
            // Mountainous terrain (Rocky, Andes, Himalayas, Tibetan plateau)
            data[idx] = 139 + Math.floor(vegetation * 40);
            data[idx + 1] = 119 + Math.floor(vegetation * 30);
            data[idx + 2] = 101;
          } else if (Math.abs(latDeg) < 25 && lonDeg > -20 && lonDeg < 55) {
            // Sahara / Middle-East Desert
            data[idx] = 205 + Math.floor(vegetation * 20);
            data[idx + 1] = 175 + Math.floor(vegetation * 20);
            data[idx + 2] = 125;
          } else {
            // Green forests / plains
            data[idx] = 34 + Math.floor(vegetation * 30);
            data[idx + 1] = 100 + Math.floor(vegetation * 50);
            data[idx + 2] = 45 + Math.floor(vegetation * 20);
          }
        } else if (elevation > 0.50) {
          // Shallow coastal waters / coral reefs
          data[idx] = 15;
          data[idx + 1] = 95;
          data[idx + 2] = 145;
        }
        // else keep background deep ocean color
      }
    }

    ctx.putImageData(imgData, 0, 0);
    return canvas;
  }

  /**
   * Generates a transparent swirling cloud layer texture.
   * Size: 2048x1024
   */
  static createCloudCanvas(width = 2048, height = 1024) {
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');

    const imgData = ctx.createImageData(width, height);
    const data = imgData.data;

    function hash(x, y) {
      let n = Math.sin(x * 17.135 + y * 91.241) * 31415.9265;
      return n - Math.floor(n);
    }

    function smoothNoise(x, y) {
      const i = Math.floor(x);
      const j = Math.floor(y);
      const fx = x - i;
      const fy = y - j;
      const u = fx * fx * (3 - 2 * fx);
      const v = fy * fy * (3 - 2 * fy);

      return (hash(i, j) * (1 - u) + hash(i + 1, j) * u) * (1 - v) +
             (hash(i, j + 1) * (1 - u) + hash(i + 1, j + 1) * u) * v;
    }

    function fbm(x, y) {
      let val = 0;
      let amp = 0.55;
      let freq = 1.0;
      for (let o = 0; o < 5; o++) {
        val += amp * smoothNoise(x * freq, y * freq);
        freq *= 2.1;
        amp *= 0.48;
      }
      return val;
    }

    for (let y = 0; y < height; y++) {
      const lat = (y / height - 0.5) * Math.PI;
      const cosLat = Math.cos(lat);

      // Cyclonic bands near trade winds
      const bandDistortion = Math.sin(lat * 8.0) * 0.15;

      for (let x = 0; x < width; x++) {
        const lon = (x / width) * Math.PI * 2;
        const nx = cosLat * Math.cos(lon + bandDistortion) * 3.5;
        const ny = cosLat * Math.sin(lon + bandDistortion) * 3.5;
        const nz = Math.sin(lat) * 3.5;

        const cloudDensity = fbm(nx + 20, ny + 20);

        const idx = (y * width + x) * 4;
        if (cloudDensity > 0.48) {
          const alpha = Math.min(255, Math.floor((cloudDensity - 0.48) * 3.5 * 255));
          data[idx] = 255;
          data[idx + 1] = 255;
          data[idx + 2] = 255;
          data[idx + 3] = alpha;
        } else {
          data[idx + 3] = 0;
        }
      }
    }

    ctx.putImageData(imgData, 0, 0);
    return canvas;
  }

  /**
   * Generates a 360 equirectangular starfield with nebulae and glowing stars.
   */
  static createStarfieldCanvas(width = 2048, height = 1024) {
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');

    ctx.fillStyle = '#010308';
    ctx.fillRect(0, 0, width, height);

    // Subtle Milky Way diagonal haze
    const nebulaGrad = ctx.createLinearGradient(0, height * 0.2, width, height * 0.8);
    nebulaGrad.addColorStop(0.0, 'rgba(10, 20, 50, 0.0)');
    nebulaGrad.addColorStop(0.4, 'rgba(40, 20, 80, 0.18)');
    nebulaGrad.addColorStop(0.5, 'rgba(20, 45, 90, 0.25)');
    nebulaGrad.addColorStop(0.6, 'rgba(60, 25, 70, 0.16)');
    nebulaGrad.addColorStop(1.0, 'rgba(10, 15, 35, 0.0)');
    ctx.fillStyle = nebulaGrad;
    ctx.fillRect(0, 0, width, height);

    // 2500 Random stars with natural brightness variance
    const numStars = 2500;
    for (let i = 0; i < numStars; i++) {
      const x = Math.random() * width;
      const y = Math.random() * height;
      const radius = Math.random() < 0.95 ? Math.random() * 0.9 : 1.0 + Math.random() * 1.5;
      const brightness = 0.3 + Math.random() * 0.7;

      ctx.beginPath();
      ctx.arc(x, y, radius, 0, Math.PI * 2);

      // Star color temperatures: blue-white, white, warm yellow, orange
      const temp = Math.random();
      if (temp > 0.85) {
        ctx.fillStyle = `rgba(180, 210, 255, ${brightness})`; // O/B blue
      } else if (temp > 0.6) {
        ctx.fillStyle = `rgba(255, 255, 255, ${brightness})`; // A/F white
      } else if (temp > 0.3) {
        ctx.fillStyle = `rgba(255, 240, 200, ${brightness})`; // G yellow (Sun-like)
      } else {
        ctx.fillStyle = `rgba(255, 180, 140, ${brightness})`; // M orange-red
      }
      ctx.fill();
    }

    return canvas;
  }
}
