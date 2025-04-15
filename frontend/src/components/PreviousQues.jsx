import React from 'react'
import style from './css/PreviousQues.module.css'

function PreviousQues() {
    return (
        <div>
            <div id={style.preQuesBox}>
                <p id={style.preQuesionsLabel}>Previous Quesions : </p>
                <p className={style.preQues}>1. What is the Virtual DOM in React?</p>
                <p className={style.preQues}>2. What are React Fragments, and why are they used?</p>
                <p className={style.preQues}>3. What is the difference between Props and State?</p>
                <p className={style.preQues}>4. What does the useEffect Hook do?</p>
                <p className={style.preQues}>5. What is the significance of the key prop in lists?</p>
            </div>
        </div>
    )
}

export default PreviousQues