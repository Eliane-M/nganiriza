import React from 'react';
import '../assets/css/landingPage/landing.css';
import { Link } from 'react-router-dom';
import myImage from '../assets/images/hero-section.jpg'


document.documentElement.style.setProperty('--hero-image', `url(${myImage})`);

const NganirizaLandingPage = () => {
  return (
    <div className="nganiriza-container">
      {/* Full-width background container */}
      <div className="background-container">
        {/* Navigation */}
        <nav className="navbar">
          <div className="logo">NGANIRIZA</div>
          <nav className='nav-menu'>
            <Link to="/user-info" className="nav-link">Predictions</Link>
            {/* <Link to="/user-info" className="nav-link">About</Link> */}
            <Link to="/visualizations" className="nav-link">Visualizations</Link>
            <Link to="/data-upload" className="nav-link">Data Upload</Link>
            <Link to="/retraining" className="nav-link">Retrain</Link>
          </nav>
        </nav>

        {/* Hero Section */}
        <div className="hero-container">
          {/* Left Side - Text Content */}
          <div className="hero-text-container">
            <p className="hero-subtitle">EMPOWER YOUR JOURNEY</p>
            <h1 className="hero-title">
              Your Safe<br />
              Space for<br />
              Health
            </h1>
            <p className="hero-description">
              Explore accurate reproductive health information and connect safely 
              with professionals in a judgment-free environment.
            </p>
            <Link to="/user-info">
              <button className="hero-cta-button">GET STARTED</button>
            </Link>
          </div>

          {/* Right Side - Circular Image */}
          <div className="hero-image-container">
            <div className="hero-circle-image">
              <img 
                src="" 
                alt="" 
                className="hero-image"
              />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default NganirizaLandingPage;