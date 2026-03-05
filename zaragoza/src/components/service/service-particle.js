import React, { useState } from 'react'
import Modal from 'react-modal';

export default function Service({ ActiveIndex }) {

    const [isOpen7, setIsOpen7] = useState(false);
    const [modalContent, setModalContent] = useState({});

    function toggleModalFour() {
        setIsOpen7(!isOpen7);
    }
    const service = [
        {
            icon: "🔬",
            title: "Precision Spectroscopy",
            text: "Sub-wavenumber molecular fingerprinting with unprecedented temporal resolution across the full vibrational spectrum.",
            text1: "Our technology enables spectroscopic measurements with temporal resolution that was previously considered physically impossible. By operating at timescales 120 orders of magnitude beyond the Planck time, we achieve sub-wavenumber accuracy across all vibrational modes.",
            text2: "Validated against 10 vibrational modes of water with less than 0.03% average error and zero adjustable parameters.",
            text3: "Applications include analytical chemistry, materials characterization, and forensic science."
        },
        {
            icon: "⚛️",
            title: "Molecular Dynamics",
            text: "Real-time bond-level temporal tracking at timescales inaccessible to any existing instrumentation.",
            text1: "Monitor molecular dynamics at temporal resolutions that capture the fastest chemical processes. Our framework provides direct access to bond-level events without the temporal averaging inherent in conventional approaches.",
            text2: "The universal scaling law (R²=1.000) ensures consistent performance across 13 orders of magnitude in frequency.",
            text3: "Ideal for studying reaction mechanisms, protein folding dynamics, and energy transfer processes."
        },
        {
            icon: "🧪",
            title: "Catalytic Monitoring",
            text: "Reaction pathway resolution at molecular timescales, enabling real-time observation of catalytic mechanisms.",
            text1: "Resolve individual steps in catalytic reaction pathways with temporal precision previously impossible. Our technology captures the transient intermediates and transition states that determine catalytic efficiency.",
            text2: "Zero adjustable parameters means the measurements are truly ab initio — no calibration against known results.",
            text3: "Applications span heterogeneous catalysis, enzyme kinetics, and industrial process optimization."
        },
        {
            icon: "💊",
            title: "Pharmaceutical Analysis",
            text: "Drug-target interaction timing with precision that enables new approaches to pharmacological design.",
            text1: "Measure the temporal dynamics of drug-receptor binding at molecular timescales. Our resolution enables the direct observation of binding kinetics, conformational changes, and allosteric effects.",
            text2: "This temporal precision opens new avenues for rational drug design and personalized medicine.",
            text3: "Applicable to high-throughput screening, ADMET profiling, and structure-activity relationship studies."
        },
        {
            icon: "🔩",
            title: "Materials Science",
            text: "Structural transition dynamics at timescales that reveal fundamental material behavior.",
            text1: "Investigate phase transitions, defect dynamics, and structural transformations at their native timescales. Our technology captures the fastest processes in materials science without temporal blurring.",
            text2: "The framework validates across the entire electromagnetic spectrum, from microwave to ultraviolet frequencies.",
            text3: "Applications include semiconductor characterization, battery materials research, and advanced manufacturing."
        },
        {
            icon: "🔭",
            title: "Fundamental Research",
            text: "Probing timescales beyond current instrumentation, opening new frontiers in physics and chemistry.",
            text1: "Access temporal regimes that have been theoretically inaccessible since the establishment of the Planck scale. Our framework provides a new experimental window into fundamental physical processes.",
            text2: "The enhancement of 10^120.95 beyond the Planck time represents a qualitative shift in measurement capability.",
            text3: "Enables new tests of quantum mechanics, investigations of ultrafast phenomena, and exploration of the boundary between quantum and classical regimes."
        }
    ]
    return (
        <>
            {/* <!-- APPLICATIONS --> */}
            <div className={ActiveIndex === 7 ? "cavani_tm_section active animated flipInX" : "cavani_tm_section hidden animated flipOutX"} id="news_">
            <div className="section_inner">
                    <div className="cavani_tm_service">
                        <div className="cavani_tm_title">
                            <span>Applications</span>
                        </div>
                        <div className="service_list">
                            <ul>
                                {service.map((item, i) => (
                                    <li key={i}>
                                        <div className="list_inner" onClick={() => { setModalContent(item); toggleModalFour(); }}>
                                            <span style={{fontSize: '48px', display: 'block', marginBottom: '20px'}}>{item.icon}</span>
                                            <h3 className="title">{item.title}</h3>
                                            <p className="text">{item.text}</p>
                                            <a className="cavani_tm_full_link" href="#" onClick={(e) => { e.preventDefault(); setModalContent(item); }} />
                                        </div>
                                    </li>
                                ))}
                            </ul>
                        </div>
                    </div>
                </div>

            </div>
            {/* <!-- APPLICATIONS --> */}

            {modalContent && (
                <Modal
                    isOpen={isOpen7}
                    onRequestClose={toggleModalFour}
                    contentLabel="My dialog"
                    className="mymodal"
                    overlayClassName="myoverlay"
                    closeTimeoutMS={300}
                    openTimeoutMS={300}
                >
                    <div className="cavani_tm_modalbox opened">
                        <div className="box_inner">
                            <div className="close" onClick={toggleModalFour} >
                                <a href="#"><i className="icon-cancel"></i></a>
                            </div>
                            <div className="description_wrap">
                                <div className="service_popup_informations">
                                    <div className="details" style={{marginBottom: '25px'}}>
                                        <span style={{fontSize: '48px'}}>{modalContent.icon}</span>
                                        <h3>{modalContent.title}</h3>
                                    </div>
                                    <div className="descriptions">
                                        <p>{modalContent.text1}</p>
                                        <p>{modalContent.text2}</p>
                                        <p>{modalContent.text3}</p>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </Modal>
            )}
        </>
    )
}
