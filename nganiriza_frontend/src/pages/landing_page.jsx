import React from 'react';
import { ChevronRight } from 'lucide-react';
import '../assets/css/landingPage/landing.css';

const NganirizaLandingPage = () => {
  return (
    <div className="nganiriza-container">
      {/* Navigation */}
      <nav className="navbar">
        <div className="logo">NGANIRIZA</div>
        {/* <div className="nav-menu">
          {/* <a href="#" className="nav-link">Home</a>
          <a href="#" className="nav-link">About</a>
          <a href="#" className="nav-link">Services</a>
          <a href="#" className="nav-link">Blog</a>
          <a href="#" className="nav-link">Contact</a> 
          <a href="#" className="cta-button">
            GET STARTED
          </a>
        </div> */}
      </nav>

      {/* Hero Section */}
      <div className="hero-container">
        {/* Left Side - Text Content */}
        <div className="hero-text-container">
          <p className="hero-subtitle">EMPOWER YOUR JOURNEY</p>
          <h1 className="hero-title">
            Your Safe Space for Health
          </h1>
          <p className="hero-description">
            Explore accurate reproductive health information and connect safely 
            with professionals in a judgment-free environment.
          </p>
          <a href="#" className="hero-cta-button">
            GET STARTED 
            
          </a>
        </div>

        {/* Right Side - Image */}
        <div className="hero-image-container">
          <div 
            className="hero-background-image"
            style={{
              backgroundImage: 'url("../../images/hero_section.png")', 
            }}
          />
          <div className="hero-circle-image">
            <img 
              src="/api/placeholder/300/300" 
              alt="Young person" 
              className="hero-image"
            />
          </div>
        </div>
      </div>
    </div>
  );
};

export default NganirizaLandingPage;