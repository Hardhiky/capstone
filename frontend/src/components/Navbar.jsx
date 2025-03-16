import React from 'react'
import style from './css/Navbar.module.css'
import { useNavigate } from 'react-router-dom';


function Navbar() {

    const navigate = useNavigate();
    return (
        <div>
            <div id={style.navbar}>
                <nav className="navbar navbar-expand-lg navbar-dark" id={style.navBar}>
                    <a className="" id={style.title} href="#">Ai Interview bot</a>
                    <button id='navBtn' className="navbar-toggler" type="button" data-toggle="collapse" data-target="#navbarSupportedContent" aria-controls="navbarSupportedContent" aria-expanded="false" aria-label="Toggle navigation">
                        <span className="navbar-toggler-icon"></span>
                    </button>

                    <div className="collapse navbar-collapse" id="navbarSupportedContent">
                        <ul className="navbar-nav mr-auto" id={style.navui}>
                            <li className="nav-item active">
                                <a className="nav-link" id={style.navItem} href="#">Home <span className="sr-only">(current)</span></a>
                            </li>
                            <li className="nav-item">
                                <a className="nav-link" id={style.navItem}>Profile</a>
                            </li>
                            <li className="nav-item">
                                <a className="nav-link" id={style.navItem}>About</a>
                            </li>
                            <li className="nav-item" onClick={()=>{navigate("/")}}>
                                <a className="nav-link" id={style.navItem}>Logout</a>
                            </li>
                        </ul>
                    </div>
                </nav>
            </div>
        </div>
    )
}

export default Navbar