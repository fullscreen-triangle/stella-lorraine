import React from 'react'

export default function Mobilemenu({isToggled, handleOnClick}) {
    return (
        <>
            {/* MOBILE MENU */}
            <div className={!isToggled ? "cavani_tm_mobile_menu" :  "cavani_tm_mobile_menu opened"} >
                <div className="inner">
                    <div className="wrapper">
                        <div className="menu_list">
                            <ul className="transition_link">
                                <li onClick={() => handleOnClick(0)}><a href="#home">Home</a></li>
                                <li onClick={() => handleOnClick(1)}><a href="#about">The Opportunity</a></li>
                                <li onClick={() => handleOnClick(2)}><a href="#portfolio">Validation</a></li>
                                <li onClick={() => handleOnClick(7)}><a href="#service">Applications</a></li>
                                <li onClick={() => handleOnClick(4)}><a href="#contact">Contact</a></li>
                            </ul>
                        </div>
                        <div className="copyright">
                            <p>&copy; 2026 Stella Lorraine</p>
                        </div>
                    </div>
                </div>
            </div>
            {/* /MOBILE MENU */}
        </>
    )
}
