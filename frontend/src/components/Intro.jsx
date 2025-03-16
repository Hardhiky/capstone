import React from "react";
import { useNavigate } from 'react-router-dom';
import style from "./css/Intro.module.css";

const Home = () => {
    // navigate inclution
    const navigate = useNavigate();

    const nextPage = () => {
        navigate("/uploadpdf");
        return 0;
    }

    return (
        <div id={style.main}>
            <div id={style.mainBox}>
                <div id={style.smallBox1}>
                    <p id={style.title}>AI Interview</p>
                    <p id={style.subTitle}>Ace Your Interviews with AI!</p>
                </div>
                <div id={style.smallBox}>
                    <p id={style.desc}>Prepare smarter with our AI-powered interview platform. Get personalized questions, real-time feedback, and expert tips to boost your confidence.</p>
                    <p id={style.desc}>Upload your resume to receive a comprehensive analysis of your skills, personalized interview questions, and expert insights. Our AI-driven platform is meticulously designed to enhance your strengths and help you confidently navigate every stage of your interview journey.</p>
                </div>
            </div>
            <button onClick={nextPage} id={style.next}>next</button>

        </div>
    )
}
export default Home;