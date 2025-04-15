import React, { useState, useEffect } from 'react';
import style from './css/Homepage.module.css';
import Navbar from './Navbar';
import PreviousQues from './PreviousQues.jsx'

function Homepage() {
  // question object
  const [question, setQuestion]=useState("6. What are React Hooks?")
    // user answer object
    const [userAnswer, setUserAnswer]=useState("")
  // ai answer display stastus
  const [answerDisplayStatus, setAnswerDisplayStatus] = useState("none")
  // ai answer object
  const [text, setAiAnswer] = useState("React Hooks let functional components manage state and side effects without class components. Key hooks include useState() for state, useEffect() for side effects, useContext() for global state, and useRef() for DOM manipulation. They make React code cleaner, more efficient, and easier to maintain.")


  // function to read out the ai answer
  const speak = () => {
    if (!text) return;
    const speech = new SpeechSynthesisUtterance(text);
    speech.lang = "en-US";
    speech.rate = 1;
    speech.pitch = 1.3;
    speech.volume=1;
    console.log(window.speechSynthesis.getVoices());
    window.speechSynthesis.speak(speech);
  };

  return (
    <div id={style.mainBox}>
      <Navbar />
      <div style={{ display: "flex", flexDirection: "row", justifyContent: "space-around" }}>
        <div id={style.questionMainBox}>
          <p id={style.question}>{question}</p>
          <div id={style.answerBox}>
            <p id={style.aiAnswerLabel}>Your Answer : </p>
            <ion-icon name="mic-outline" id={style.microphoneLogo}></ion-icon>
            <textarea id={style.answer} placeholder='Enter your answer here...' onChange={(e)=>{setUserAnswer(e.target.value)}}>{userAnswer}</textarea>
          </div>
          <input type='submit' id={style.submitAnswerButton} onClick={() => { setAnswerDisplayStatus("block") }} value="submit" />

          <div id={style.answerBox} style={{ display: answerDisplayStatus }}>
            <p id={style.aiAnswerLabel}>Ai Answer : </p>
            <ion-icon name="volume-high" onClick={speak} id={style.microphoneLogo}></ion-icon>
            <p id={style.aiAnswer}>{text}</p>
          </div>
        </div>

        <PreviousQues />
      </div>
    </div >
  )
}
export default Homepage;