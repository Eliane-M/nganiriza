import React from 'react';
import '../assets/css/landingPage/landing.css';
import { Link } from 'react-router-dom';

const NganirizaLandingPage = () => {
  return (
    <div className="nganiriza-container">
      {/* Full-width background container */}
      <div className="background-container">
        {/* Navigation */}
        <nav className="navbar">
          <div className="logo">NGANIRIZA</div>
          <div className="nav-menu">
            <a href="user-info" className="nav-link">Predictions</a>
            {/* <a href="user-info" className="nav-link">About</a> */}
            <a href="#" className="nav-link">Visualizations</a>
            <a href="data-upload" className="nav-link">Data Upload</a>
            <a href="retraining" className="nav-link">Retrain</a>
          </div>
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