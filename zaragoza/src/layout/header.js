import React from 'react'

export default function Header({handleOnClick, ActiveIndex}) {

    return (
        <>
            {/* HEADER */}
            <div className="cavani_tm_header">
                <div className="logo">
                    <a href="#" style={{textDecoration: 'none', color: '#e8e6ef', fontFamily: 'Poppins', fontSize: '22px', fontWeight: '700', letterSpacing: '3px', textTransform: 'uppercase'}}>
                        Stella Lorraine
                    </a>
                </div>
                <div className="menu">
                    <ul className="transition_link">
                        <li onClick={() => handleOnClick(0)}><a className={ActiveIndex === 0 ? "active" : ""}>Home</a></li>
                        <li onClick={() => handleOnClick(1)}><a className={ActiveIndex === 1 ? "active" : ""}>The Opportunity</a></li>
                        <li onClick={() => handleOnClick(2)}><a className={ActiveIndex === 2 ? "active" : ""}>Validation</a></li>
                        <li onClick={() => handleOnClick(7)}><a className={ActiveIndex === 7 ? "active" : ""}>Applications</a></li>
                        <li onClick={() => handleOnClick(4)}><a className={ActiveIndex === 4 ? "active" : ""}>Contact</a></li>
                    </ul>
                </div>
            </div>
            {/* /HEADER */}

        </>
    )
}
