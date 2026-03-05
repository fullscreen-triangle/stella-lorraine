import Link from "next/link";
import React from "react";
import { LoadingTextAnimation } from "../AnimationText";

export default function HomeDefault({ ActiveIndex, handleOnClick }) {
  return (
    <>
      {/* <!-- HOME --> */}
      <div
        className={
          ActiveIndex === 0
            ? "cavani_tm_section active animated flipInX"
            : "cavani_tm_section active hidden animated flipOutX"
        }
        id="home_"
      >
        <div className="cavani_tm_home">
          <div className="content">
            <h3 className="name">Stella Lorraine</h3>
            <span className="line" style={{backgroundColor: '#6c63ff'}}></span>
            <h3 className="job">
              <LoadingTextAnimation />
            </h3>
            <p style={{color: '#8a8697', marginBottom: '35px', fontSize: '16px', maxWidth: '500px'}}>
              Temporal resolution that physics said was impossible.
            </p>
            <div className="cavani_tm_button transition_link">
              <Link href="#contact">
                <a onClick={() => handleOnClick(4)}>Request a Briefing</a>
              </Link>
            </div>
          </div>
        </div>
      </div>
      {/* <!-- HOME --> */}
    </>
  );
}
