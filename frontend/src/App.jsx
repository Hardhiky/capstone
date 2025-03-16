import React from "react";
import { BrowserRouter, Router, Route, Routes } from "react-router-dom";
import Intro from "./components/Intro";
import UploadPDF from '../src/components/UploadPDF'
import Home from "./components/Homepage";


import './components/css/App.module.css'

function App(){
  return(
    <BrowserRouter>
      <br/>
      <Routes>
        <Route index element={<Intro/>}/>
        <Route path="/uploadpdf" element={<UploadPDF/>}/>
        <Route path="/home" element={<Home/>}/>
      </Routes>
    </BrowserRouter>
  )
}
export default App;