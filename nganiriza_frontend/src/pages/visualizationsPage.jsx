import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import '../assets/css/visualizationsPage/visualizationsPage.css';

const VisualizationsPage = () => {
  const [visualizations, setVisualizations] = useState({
    education: { plot1: null, plot2: null },
    contraceptive: { plot1: null, plot2: null },
    income: { plot1: null, plot2: null },
    healthcare: { plot1: null, plot2: null },
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Fetch visualizations from the backend
  const fetchVisualizations = async (plotType) => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch(`http://localhost:8000/visualizations/${plotType}`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || `Failed to fetch ${plotType} visualization`);
      }

      const data = await response.json();
      setVisualizations((prev) => ({
        ...prev,
        [plotType]: { plot1: data.plot1, plot2: data.plot2 },
      }));
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  // Load initial visualizations (optional: trigger on mount)
  useEffect(() => {
    // Uncomment to fetch all on page load
    // fetchVisualizations('education');
    // fetchVisualizations('contraceptive');
    // fetchVisualizations('income');
    // fetchVisualizations('healthcare');
  }, []);

  // Handle button clicks to fetch specific visualizations
  const handleFetchVisualization = (plotType) => {
    fetchVisualizations(plotType);
  };

  return (
    <div className="visualizations-container">
      <header className="header">
        <h1><a href="/">NGANIRIZA</a></h1>
        <nav>
            <Link to="/user-info" className="nav-link">Predictions</Link>
            <Link to="/visualizations" className="nav-link">Visualizations</Link>
            <Link to="/data-upload" className="nav-link">Data Upload</Link>
            <Link to="/retraining" className="nav-link">Retrain</Link>
        </nav>
      </header>

      <div className="visualizations-wrapper">
        <h2>Data Visualizations</h2>
        <p>Explore the relationships between key factors and risk categories.</p>

        <div className="button-group">
          <button
            onClick={() => handleFetchVisualization('education')}
            disabled={loading}
          >
            {loading ? 'Loading...' : 'Education Level'}
          </button>
          <button
            onClick={() => handleFetchVisualization('contraceptive')}
            disabled={loading}
          >
            {loading ? 'Loading...' : 'Contraceptive Use'}
          </button>
          <button
            onClick={() => handleFetchVisualization('income')}
            disabled={loading}
          >
            {loading ? 'Loading...' : 'Family Income'}
          </button>
          <button
            onClick={() => handleFetchVisualization('healthcare')}
            disabled={loading}
          >
            {loading ? 'Loading...' : 'Healthcare Access'}
          </button>
        </div>

        {error && <p className="error-message">{error}</p>}

        <div className="visualizations-grid">
          {/* Education Level Visualizations */}
          <div className="visualization-section">
            <h3>Education Level</h3>
            {visualizations.education.plot1 ? (
              <img src={visualizations.education.plot1} alt="Education Plot 1" />
            ) : (
              <p>No visualization available. Click the button to load.</p>
            )}
            {visualizations.education.plot2 && (
              <img src={visualizations.education.plot2} alt="Education Plot 2" />
            )}
          </div>

          {/* Contraceptive Use Visualizations */}
          <div className="visualization-section">
            <h3>Contraceptive Use</h3>
            {visualizations.contraceptive.plot1 ? (
              <img src={visualizations.contraceptive.plot1} alt="Contraceptive Plot 1" />
            ) : (
              <p>No visualization available. Click the button to load.</p>
            )}
            {visualizations.contraceptive.plot2 && (
              <img src={visualizations.contraceptive.plot2} alt="Contraceptive Plot 2" />
            )}
          </div>

          {/* Family Income Visualizations */}
          <div className="visualization-section">
            <h3>Family Income</h3>
            {visualizations.income.plot1 ? (
              <img src={visualizations.income.plot1} alt="Income Plot 1" />
            ) : (
              <p>No visualization available. Click the button to load.</p>
            )}
            {visualizations.income.plot2 && (
              <img src={visualizations.income.plot2} alt="Income Plot 2" />
            )}
          </div>

          {/* Healthcare Access Visualizations */}
          <div className="visualization-section">
            <h3>Healthcare Access</h3>
            {visualizations.healthcare.plot1 ? (
              <img src={visualizations.healthcare.plot1} alt="Healthcare Plot 1" />
            ) : (
              <p>No visualization available. Click the button to load.</p>
            )}
            {visualizations.healthcare.plot2 && (
              <img src={visualizations.healthcare.plot2} alt="Healthcare Plot 2" />
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default VisualizationsPage;