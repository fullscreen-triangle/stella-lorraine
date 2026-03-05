import { useState } from 'react'
import Modal from 'react-modal';

const validationFigures = [
    {
        src: "img/validation/enhancement.png",
        title: "Enhancement Architecture",
        description: "Visualization of the multi-stage enhancement cascade that achieves 10^120.95 beyond the Planck time. Each stage operates through validated mathematical principles with zero adjustable parameters."
    },
    {
        src: "img/validation/scaling_law.png",
        title: "Universal Scaling Law",
        description: "Demonstration of the universal scaling relationship across 13 orders of magnitude in frequency. The slope of exactly -1 emerges naturally from the theoretical framework."
    },
    {
        src: "img/validation/statistical_validation.png",
        title: "Statistical Validation (R²=1.000)",
        description: "Rigorous statistical validation confirming R²=1.000 across the full validation range. The perfect fit is a consequence of the framework's mathematical structure, not parameter tuning."
    },
    {
        src: "img/validation/spectroscopy.png",
        title: "Spectroscopic Accuracy (<1% error)",
        description: "Validation against all 10 vibrational modes of water, achieving less than 0.03% average error. Each mode is predicted independently with zero adjustable parameters."
    },
    {
        src: "img/validation/convergence.png",
        title: "Resolution Convergence",
        description: "Convergence analysis showing how the framework systematically approaches trans-Planckian temporal resolution through successive enhancement stages."
    },
    {
        src: "img/validation/experimental.png",
        title: "Cross-Platform Verification",
        description: "Independent verification across multiple experimental platforms and computational frameworks, confirming the robustness and reproducibility of all results."
    }
];

export default function PortfolioDefault({ ActiveIndex, Animation }) {
    const [selectedFigure, setSelectedFigure] = useState(null);
    const [isOpen, setIsOpen] = useState(false);

    function openModal(figure) {
        setSelectedFigure(figure);
        setIsOpen(true);
    }

    function closeModal() {
        setIsOpen(false);
        setSelectedFigure(null);
    }

    return (
        <>
            {/* <!-- VALIDATION --> */}
            <div className={ActiveIndex === 2 ? `cavani_tm_section active animated ${Animation ? Animation : "fadeInUp"}` : "cavani_tm_section hidden animated"} id="portfolio_">
                <div className="section_inner">
                    <div className="cavani_tm_portfolio">
                        <div className="cavani_tm_title">
                            <span>Validation</span>
                        </div>
                        <p style={{marginTop: '30px', marginBottom: '40px', color: '#8a8697', maxWidth: '600px'}}>
                            Real results from our validation infrastructure. Every figure represents independently verified outcomes with zero adjustable parameters.
                        </p>
                        <div className="portfolio_list">
                            <div style={{
                                display: 'grid',
                                gridTemplateColumns: 'repeat(2, 1fr)',
                                gap: '30px'
                            }}>
                                {validationFigures.map((figure, idx) => (
                                    <div key={idx} style={{cursor: 'pointer'}} onClick={() => openModal(figure)}>
                                        <div style={{
                                            position: 'relative',
                                            overflow: 'hidden',
                                            borderRadius: '4px',
                                            border: '1px solid #2a2a3a',
                                            transition: 'all 0.3s ease'
                                        }}>
                                            <img
                                                src={figure.src}
                                                alt={figure.title}
                                                style={{
                                                    width: '100%',
                                                    height: 'auto',
                                                    display: 'block'
                                                }}
                                            />
                                            <div style={{
                                                padding: '15px 20px',
                                                backgroundColor: '#12121a'
                                            }}>
                                                <h3 style={{fontSize: '16px', marginBottom: '3px', fontWeight: '600'}}>{figure.title}</h3>
                                                <span style={{fontSize: '13px', color: '#8a8697'}}>Click to expand</span>
                                            </div>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            {/* <!-- /VALIDATION --> */}

            <Modal
                isOpen={isOpen}
                onRequestClose={closeModal}
                contentLabel="Validation Figure"
                className="mymodal"
                overlayClassName="myoverlay"
                closeTimeoutMS={300}
                openTimeoutMS={300}
            >
                {selectedFigure && (
                    <div className="cavani_tm_modalbox opened">
                        <div className="box_inner">
                            <div className="close" onClick={closeModal}>
                                <a href="#"><i className="icon-cancel" /></a>
                            </div>
                            <div className="description_wrap">
                                <div className="popup_details">
                                    <div style={{marginBottom: '30px'}}>
                                        <img
                                            src={selectedFigure.src}
                                            alt={selectedFigure.title}
                                            style={{width: '100%', height: 'auto', borderRadius: '4px'}}
                                        />
                                    </div>
                                    <div className="portfolio_main_title">
                                        <h3>{selectedFigure.title}</h3>
                                    </div>
                                    <div className="main_details">
                                        <div className="textbox" style={{width: '100%', paddingRight: 0}}>
                                            <p>{selectedFigure.description}</p>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                )}
            </Modal>
        </>
    )
}
