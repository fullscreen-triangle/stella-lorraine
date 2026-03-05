import React from 'react'
import ProgressBar from '../progressBar';
import { CircularProgressbar } from "react-circular-progressbar";
import "react-circular-progressbar/dist/styles.css";

const progressBarData = [
    { bgcolor: "#6c63ff", completed: 100, title: 'Resolution Enhancement: 10^121×' },
    { bgcolor: "#00d4aa", completed: 97, title: 'Spectroscopic Accuracy: 99.97%' },
    { bgcolor: "#6c63ff", completed: 100, title: 'Frequency Validation Range: 13 Orders' },
    { bgcolor: "#2a2a3a", completed: 0, title: 'Adjustable Parameters: Zero' },
];

const circleProgressData = [
    { language: 'Modes Validated', progress: 100 },
    { language: 'Subsystems Pass', progress: 100 },
    { language: 'Error Rate', progress: 3 },
];

export default function AboutDefault({ ActiveIndex }) {
    return (
        <>
            {/* <!-- ABOUT --> */}
            <div className={ActiveIndex === 1 ? "cavani_tm_section active animated flipInX" : "cavani_tm_section active hidden animated flipOutX"} id="about_">
            <div className="section_inner">
                    <div className="cavani_tm_about">
                        <div className="biography">
                            <div className="cavani_tm_title">
                                <span>The Opportunity</span>
                            </div>
                            <div className="wrapper">
                                <div className="left">
                                    <p>For over a century, the <strong style={{color: '#6c63ff'}}>Planck time</strong> has stood as physics&apos; most stringent barrier &mdash; the shortest measurable interval at 10<sup>-44</sup> seconds.</p>
                                    <p>Every attempt to break it fails. <strong style={{color: '#e8e6ef'}}>We found a different path.</strong></p>
                                    <p style={{marginTop: '20px', color: '#8a8697'}}>Our proprietary framework achieves temporal resolution at <strong style={{color: '#00d4aa'}}>~10<sup>-165</sup> seconds</strong> &mdash; validated across 13 orders of magnitude with zero adjustable parameters.</p>
                                </div>
                                <div className="right">
                                    <ul>
                                        <li><span className="first">Resolution:</span><span className="second" style={{color: '#00d4aa'}}>~10⁻¹⁶⁵ s</span></li>
                                        <li><span className="first">Enhancement:</span><span className="second" style={{color: '#6c63ff'}}>10^120.95×</span></li>
                                        <li><span className="first">Accuracy:</span><span className="second" style={{color: '#00d4aa'}}>&lt;0.03% error</span></li>
                                        <li><span className="first">Validation:</span><span className="second">R² = 1.000</span></li>
                                        <li><span className="first">Parameters:</span><span className="second" style={{color: '#6c63ff'}}>Zero</span></li>
                                        <li><span className="first">Range:</span><span className="second">13 orders of magnitude</span></li>
                                    </ul>
                                </div>
                            </div>
                        </div>
                        <div className="skills">
                            <div className="wrapper">
                                <div className="programming">
                                    <div className="cavani_tm_title">
                                        <span>Key Metrics</span>
                                    </div>
                                    <div className="cavani_progress">
                                        {progressBarData.map((item, idx) => (
                                            <ProgressBar key={idx} bgcolor={item.bgcolor} completed={item.completed} title={item.title} />
                                        ))}
                                    </div>
                                </div>
                                <div className="language">
                                    <div className="cavani_tm_title">
                                        <span>Validation</span>
                                    </div>
                                    <div className="circular_progress_bar">
                                        <div className='circle_holder'>
                                            {circleProgressData.map((item, idx) => (
                                                <div key={idx}>
                                                    <div className="list_inner">
                                                        <CircularProgressbar
                                                            value={item.progress}
                                                            text={`${item.progress}%`}
                                                            strokeWidth={3}
                                                            styles={{
                                                                path: { stroke: item.progress === 100 ? '#00d4aa' : '#6c63ff' },
                                                                text: { fill: '#e8e6ef', fontSize: '16px' },
                                                                trail: { stroke: '#2a2a3a' }
                                                            }}
                                                            className={"list_inner"}
                                                        />
                                                        <div className="title"><span>{item.language}</span></div>
                                                    </div>
                                                </div>
                                            ))}
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div className="resume">
                            <div className="wrapper">
                                <div className="education">
                                    <div className="cavani_tm_title">
                                        <span>Validated Capabilities</span>
                                    </div>
                                    <div className="list">
                                        <div className="univ">
                                            <ul>
                                                <li>
                                                    <div className="list_inner">
                                                        <div className="time">
                                                            <span style={{backgroundColor: 'rgba(108, 99, 255, 0.15)', color: '#6c63ff'}}>Resolution</span>
                                                        </div>
                                                        <div className="place">
                                                            <h3>~10⁻¹⁶⁵ seconds</h3>
                                                            <span>Temporal resolution</span>
                                                        </div>
                                                    </div>
                                                </li>
                                                <li>
                                                    <div className="list_inner">
                                                        <div className="time">
                                                            <span style={{backgroundColor: 'rgba(108, 99, 255, 0.15)', color: '#6c63ff'}}>Enhancement</span>
                                                        </div>
                                                        <div className="place">
                                                            <h3>10^120.95 ×</h3>
                                                            <span>Beyond Planck time</span>
                                                        </div>
                                                    </div>
                                                </li>
                                                <li>
                                                    <div className="list_inner">
                                                        <div className="time">
                                                            <span style={{backgroundColor: 'rgba(0, 212, 170, 0.15)', color: '#00d4aa'}}>Accuracy</span>
                                                        </div>
                                                        <div className="place">
                                                            <h3>&lt;0.03% spectroscopic error</h3>
                                                            <span>10/10 vibrational modes validated</span>
                                                        </div>
                                                    </div>
                                                </li>
                                            </ul>
                                        </div>
                                    </div>
                                </div>
                                <div className="experience">
                                    <div className="cavani_tm_title">
                                        <span>Applications</span>
                                    </div>
                                    <div className="list">
                                        <div className="univ">
                                            <ul>
                                                <li>
                                                    <div className="list_inner">
                                                        <div className="time">
                                                            <span style={{backgroundColor: 'rgba(108, 99, 255, 0.15)', color: '#6c63ff'}}>Spectroscopy</span>
                                                        </div>
                                                        <div className="place">
                                                            <h3>Precision Spectroscopy</h3>
                                                            <span>Sub-wavenumber molecular fingerprinting</span>
                                                        </div>
                                                    </div>
                                                </li>
                                                <li>
                                                    <div className="list_inner">
                                                        <div className="time">
                                                            <span style={{backgroundColor: 'rgba(0, 212, 170, 0.15)', color: '#00d4aa'}}>Dynamics</span>
                                                        </div>
                                                        <div className="place">
                                                            <h3>Molecular Dynamics</h3>
                                                            <span>Real-time bond-level temporal tracking</span>
                                                        </div>
                                                    </div>
                                                </li>
                                                <li>
                                                    <div className="list_inner">
                                                        <div className="time">
                                                            <span style={{backgroundColor: 'rgba(108, 99, 255, 0.15)', color: '#6c63ff'}}>Pharma</span>
                                                        </div>
                                                        <div className="place">
                                                            <h3>Pharmaceutical Analysis</h3>
                                                            <span>Drug-target interaction timing</span>
                                                        </div>
                                                    </div>
                                                </li>
                                            </ul>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            {/* <!-- ABOUT --> */}
        </>
    )
}
