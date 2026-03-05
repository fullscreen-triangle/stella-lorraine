import React, { useState, useEffect } from 'react'
import { customCursor } from '../../plugin/plugin';

export default function ContactDefault({ ActiveIndex }) {
    useEffect(() => {
        customCursor();
    });

    const [form, setForm] = useState({ email: "", name: "", org: "", msg: "" });
    const [active, setActive] = useState(null);
    const [error, setError] = useState(false);
    const [success, setSuccess] = useState(false);
    const onChange = (e) => {
        setForm({ ...form, [e.target.name]: e.target.value });
    };
    const { email, name, org, msg } = form;
    const onSubmit = (e) => {
        e.preventDefault();
        if (email && name && msg) {
            setSuccess(true);
            setTimeout(() => {
                setForm({ email: "", name: "", org: "", msg: "" });
                setSuccess(false);
            }, 3000);
        } else {
            setError(true);
            setTimeout(() => {
                setError(false);
            }, 2000);
        }
    };
    return (
        <>
            {/* <!-- CONTACT --> */}
            <div className={ActiveIndex === 4 ? "cavani_tm_section active animated flipInX" : "cavani_tm_section hidden animated flipOutX"} id="contact_">
            <div className="section_inner">
                    <div className="cavani_tm_contact">
                        <div className="cavani_tm_title">
                            <span>Request a Briefing</span>
                        </div>
                        <p style={{marginTop: '30px', marginBottom: '10px', color: '#8a8697', maxWidth: '550px'}}>
                            We are selectively engaging with strategic partners and investors. Tell us about your interest.
                        </p>
                        <div className="short_info">
                            <ul>
                                <li>
                                    <div className="list_inner">
                                        <i className="icon-mail-3"></i>
                                        <span><a href="mailto:contact@stellalorraine.com" style={{color: '#6c63ff'}}>contact@stellalorraine.com</a></span>
                                    </div>
                                </li>
                            </ul>
                        </div>
                        <div className="form">
                            <div className="left" style={{width: '100%', paddingRight: 0}}>
                                <div className="fields">
                                    {/* Contact Form */}
                                    <form className="contact_form" onSubmit={(e) => onSubmit(e)}>
                                        <div
                                            className="returnmessage"
                                            style={{ display: success ? "block" : "none" }}
                                        >
                                            <span className="contact_success" style={{color: '#00d4aa'}}>
                                                Your request has been received. We will be in touch shortly.
                                            </span>
                                        </div>
                                        <div
                                            className="empty_notice"
                                            style={{ display: error ? "block" : "none" }}
                                        >
                                            <span>Please fill in the required fields.</span>
                                        </div>

                                        <div className="fields">
                                            <ul>
                                                <li
                                                    className={`input_wrapper ${active === "name" || name ? "active" : ""}`}
                                                >
                                                    <input
                                                        onFocus={() => setActive("name")}
                                                        onBlur={() => setActive(null)}
                                                        onChange={(e) => onChange(e)}
                                                        value={name}
                                                        name="name"
                                                        id="name"
                                                        type="text"
                                                        placeholder="Name *"
                                                        style={{backgroundColor: 'transparent', color: '#e8e6ef'}}
                                                    />
                                                </li>
                                                <li
                                                    className={`input_wrapper ${active === "email" || email ? "active" : ""}`}
                                                >
                                                    <input
                                                        onFocus={() => setActive("email")}
                                                        onBlur={() => setActive(null)}
                                                        onChange={(e) => onChange(e)}
                                                        value={email}
                                                        name="email"
                                                        id="email"
                                                        type="email"
                                                        placeholder="Email *"
                                                        style={{backgroundColor: 'transparent', color: '#e8e6ef'}}
                                                    />
                                                </li>
                                                <li
                                                    className={`input_wrapper ${active === "org" || org ? "active" : ""}`}
                                                >
                                                    <input
                                                        onFocus={() => setActive("org")}
                                                        onBlur={() => setActive(null)}
                                                        onChange={(e) => onChange(e)}
                                                        value={org}
                                                        name="org"
                                                        id="org"
                                                        type="text"
                                                        placeholder="Organization"
                                                        style={{backgroundColor: 'transparent', color: '#e8e6ef'}}
                                                    />
                                                </li>
                                                <li
                                                    className={`last ${active === "message" || msg ? "active" : ""}`}
                                                >
                                                    <textarea
                                                        onFocus={() => setActive("message")}
                                                        onBlur={() => setActive(null)}
                                                        name="msg"
                                                        onChange={(e) => onChange(e)}
                                                        value={msg}
                                                        id="message"
                                                        placeholder="Tell us about your interest *"
                                                        style={{backgroundColor: 'transparent', color: '#e8e6ef'}}
                                                    />
                                                </li>
                                            </ul>
                                            <div className="cavani_tm_button">
                                                <input
                                                    className='a'
                                                    type="submit"
                                                    id="send_message"
                                                    value="Submit Request"
                                                />
                                            </div>
                                        </div>
                                    </form>
                                    {/* /Contact Form */}
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            {/* <!-- CONTACT --> */}
        </>
    )
}
