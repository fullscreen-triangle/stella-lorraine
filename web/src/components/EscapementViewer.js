import { useEffect, useRef } from "react";

export default function EscapementViewer() {
  const mountRef = useRef(null);

  useEffect(() => {
    let animId;
    let renderer;
    let mounted = true;

    const init = async () => {
      const THREE = await import("three");
      const { GLTFLoader } = await import(
        "three/examples/jsm/loaders/GLTFLoader.js"
      );
      const { OrbitControls } = await import(
        "three/examples/jsm/controls/OrbitControls.js"
      );

      if (!mounted || !mountRef.current) return;
      const mount = mountRef.current;
      const w = mount.clientWidth;
      const h = mount.clientHeight;

      const scene = new THREE.Scene();
      scene.background = new THREE.Color(0x1b1b1b);

      const camera = new THREE.PerspectiveCamera(40, w / h, 0.001, 200);
      camera.position.set(0, 0.3, 4);

      renderer = new THREE.WebGLRenderer({ antialias: true });
      renderer.setSize(w, h);
      renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
      renderer.physicallyCorrectLights = true;
      renderer.outputEncoding = THREE.sRGBEncoding;
      renderer.toneMapping = THREE.ACESFilmicToneMapping;
      renderer.toneMappingExposure = 0.9;
      mount.appendChild(renderer.domElement);

      // Lighting: key + rim in brand colors
      const ambient = new THREE.AmbientLight(0xffffff, 0.4);
      scene.add(ambient);

      const key = new THREE.DirectionalLight(0xffffff, 2.5);
      key.position.set(4, 8, 5);
      scene.add(key);

      const rim = new THREE.DirectionalLight(0x58e6d9, 0.6);
      rim.position.set(-5, -2, 3);
      scene.add(rim);

      const back = new THREE.DirectionalLight(0xb63e96, 0.3);
      back.position.set(0, -8, -5);
      scene.add(back);

      const controls = new OrbitControls(camera, renderer.domElement);
      controls.enableDamping = true;
      controls.dampingFactor = 0.04;
      controls.autoRotate = true;
      controls.autoRotateSpeed = 0.4;
      controls.enablePan = false;
      controls.minDistance = 1.5;
      controls.maxDistance = 12;
      controls.enableZoom = true;

      const loader = new GLTFLoader();
      loader.load(
        "/swiss_lever_escapement_mechanism.glb",
        (gltf) => {
          if (!mounted) return;
          const model = gltf.scene;

          const box = new THREE.Box3().setFromObject(model);
          const center = box.getCenter(new THREE.Vector3());
          const size = box.getSize(new THREE.Vector3());
          const maxDim = Math.max(size.x, size.y, size.z);
          const scale = 2.8 / maxDim;

          model.scale.setScalar(scale);
          model.position.sub(center.multiplyScalar(scale));

          model.traverse((child) => {
            if (child.isMesh) {
              child.material = new THREE.MeshStandardMaterial({
                color: 0xb8bfc8,
                metalness: 0.88,
                roughness: 0.22,
              });
            }
          });

          scene.add(model);
        },
        undefined,
        (err) => console.error("GLB load error:", err)
      );

      const onResize = () => {
        if (!mounted || !mountRef.current) return;
        const w = mountRef.current.clientWidth;
        const h = mountRef.current.clientHeight;
        camera.aspect = w / h;
        camera.updateProjectionMatrix();
        renderer.setSize(w, h);
      };
      window.addEventListener("resize", onResize);

      const animate = () => {
        if (!mounted) return;
        animId = requestAnimationFrame(animate);
        controls.update();
        renderer.render(scene, camera);
      };
      animate();

      return () => {
        window.removeEventListener("resize", onResize);
      };
    };

    const cleanupPromise = init();

    return () => {
      mounted = false;
      cancelAnimationFrame(animId);
      cleanupPromise.then((fn) => fn && fn());
      if (renderer) {
        renderer.dispose();
        if (
          mountRef.current &&
          renderer.domElement.parentNode === mountRef.current
        ) {
          mountRef.current.removeChild(renderer.domElement);
        }
      }
    };
  }, []);

  return (
    <div
      ref={mountRef}
      style={{ width: "100vw", height: "100vh", background: "#1b1b1b" }}
    />
  );
}
