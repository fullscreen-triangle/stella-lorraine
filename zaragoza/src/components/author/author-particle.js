import "particles.js/particles";
import React, { useEffect } from "react";
export default function AuthorDefault() {
  useEffect(() => {
    const particlesJS = window.particlesJS;
    particlesJS.load("particles-js", "particlesConfig.json", function () {
      console.log("particles loaded");
    });
  }, []);

  return (
    <>
      <div className="author_image">
        <div
          className="main"
          style={{ backgroundColor: '#0a0a0f' }}
        ></div>
        <div className="particle_wrapper">
          <div id="particles-js" />
        </div>
      </div>
    </>
  );
}
