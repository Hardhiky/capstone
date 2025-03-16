import React, { useState } from 'react';
import style from "./css/UploadPDF.module.css";
import { useNavigate } from 'react-router-dom';


function UploadPDF() {

  const navigate = useNavigate();

  const [fileName, setFileName] = useState("No file selected");
  const [file, setFile] = useState(null);

  const handleFileUpload = (e) => {
    const selFile = e.target.files[0];
    if (selFile) {
      setFile(selFile)
      setFileName(selFile.name)
      console.log(file)
    }
  }
  const toHomepage = () => {
    navigate("/home");
  }
  return (
    <div id={style.main}>
      <div id={style.container}>
        <p id={style.title}>We Extract Skills from Your Resume</p>
        <p id={style.heading}>Showcase Your Strengths: Extract Skills from Your Resume!</p>
        <p id={style.text1}>Upload the Resume</p>
        <form>
          {/* upload button */}
          <label htmlFor='file' id={style.fileInput} onChange={handleFileUpload}>
            <input type='file' id="file" accept='.pdf' hidden />
            <ion-icon id={style.uploadIcon} name="cloud-upload"></ion-icon>
          </label>
          <center>
          <p id={style.fileSelected}>{fileName}</p>

          <input type='submit' id={style.uploadBtn} value="Upload" />
          </center>
        </form>
      </div>
      <button id={style.skip} onClick={toHomepage}>wanna skip?</button>
    </div >

  )
}

export default UploadPDF