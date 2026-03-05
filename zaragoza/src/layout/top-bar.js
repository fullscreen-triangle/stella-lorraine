import React from 'react'

export default function TopBar({toggleTrueFalse, isToggled}) {

    return (
        <>
            <div className="cavani_tm_topbar">
                <div className="topbar_inner">
                    <div className="logo">
                        <a href="#" style={{textDecoration: 'none', color: '#e8e6ef', fontFamily: 'Poppins', fontSize: '18px', fontWeight: '700', letterSpacing: '2px', textTransform: 'uppercase'}}>
                            Stella Lorraine
                        </a>
                    </div>
                    <div className="trigger">
                        <div onClick={toggleTrueFalse} className={!isToggled ? "hamburger hamburger--slider" : "hamburger hamburger--slider is-active"}>
                            <div className="hamburger-box">
                                <div className="hamburger-inner"></div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </>
    )
}
