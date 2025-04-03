// Retraining.jsx
import React, { useState } from 'react';
import '../assets/css/retraining/retrainingPage.css';
import { Link } from 'react-router-dom';

const Retraining = () => {
    const [isRetraining, setIsRetraining] = useState(false);
    const [message, setMessage] = useState('');

  const handleRetrain = async () => {
    setIsRetraining(true);
    setMessage("Retraining the model...");
  
    try {
      const response = await fetch("https://nganiriza.onrender.com/retrain", {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
      });
  
      const data = await response.json();
      if (response.ok) {
        setMessage(`Retraining completed: ${data.message}`);
      } else {
        throw new Error(data.detail || "Retrain failed");
      }
    } catch (error) {
      setMessage(`Error: ${error.message}`);
    } finally {
        setIsRetraining(false);
    }
  };

  return (
    <div className="retraining-container">
      <header className="header">
        <h1><a href='/'>NGANIRIZA</a></h1>
        <nav>
            <Link to="/user-info" className="nav-link">Predictions</Link>
            {/* <Link to="/user-info" className="nav-link">About</Link> */}
            <Link to="/visualizations" className="nav-link">Visualizations</Link>
            <Link to="/data-upload" className="nav-link">Data Upload</Link>
            <Link to="/retraining" className="nav-link">Retrain</Link>
        </nav>
      </header>

      <main className="main-content">
        <h2>Upload Dataset to Retrain</h2>
        <p>This will retrain the latest uploaded dataset.</p>

        <div className="retraining-section">

          <button
            className="retrain-button"
            onClick={handleRetrain}
            disabled={isRetraining}
          >
            {isRetraining ? 'Retraining...' : 'Start Retraining'}
          </button>
          {message && <p className="upload-message">{message}</p>}
        </div>
      </main>
    </div>
  );
};

export default Retraining;
