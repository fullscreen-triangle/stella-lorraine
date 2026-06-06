import React, { Suspense } from "react";
import * as THREE from "three";
import { Canvas } from "@react-three/fiber";
import { OrbitControls, Gltf, Bounds, Center } from "@react-three/drei";
import { EffectComposer, Bloom } from "@react-three/postprocessing";

// ── 3D atom viewer (react-three-fiber + drei) ─────────────────────────────────
// The composition trajectory resolves to an element; this renders that element's
// actual GLB model. drei's <Gltf> + <Bounds>/<Center> auto-frame any model; a
// brand-tinted light rig (added in onCreated to avoid R3F intrinsic-JSX typing)
// plus a Bloom pass give the nucleus/orbital glow. Switching `src` remounts the
// model (drei caches by URL, so revisits are instant).

function setup({ scene }: { scene: THREE.Scene }) {
  scene.background = new THREE.Color("#02100e");
  scene.add(new THREE.AmbientLight(0xffffff, 0.55));
  const key = new THREE.DirectionalLight(0xffffff, 2.2); key.position.set(4, 8, 5); scene.add(key);
  const rim = new THREE.DirectionalLight(0x58e6d9, 0.7); rim.position.set(-5, -2, 3); scene.add(rim);
  const back = new THREE.DirectionalLight(0xa78bfa, 0.45); back.position.set(0, -8, -5); scene.add(back);
}

export function AtomViewer({ src }: { src: string }) {
  return (
    <Canvas
      dpr={[1, 2]}
      camera={{ position: [0, 0.3, 4.6], fov: 42 }}
      gl={{ antialias: true }}
      onCreated={setup as any}
      style={{ width: "100%", height: "100%", display: "block" }}
    >
      <Suspense fallback={null}>
        <Bounds fit clip observe margin={1.15}>
          <Center key={src}>
            <Gltf src={src} />
          </Center>
        </Bounds>
      </Suspense>
      <OrbitControls
        makeDefault
        autoRotate
        autoRotateSpeed={0.8}
        enablePan={false}
        enableDamping
        dampingFactor={0.06}
        minDistance={1.2}
        maxDistance={16}
      />
      <EffectComposer>
        <Bloom luminanceThreshold={0.3} intensity={0.5} mipmapBlur />
      </EffectComposer>
    </Canvas>
  );
}
