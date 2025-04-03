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
  const [selectedPlotType, setSelectedPlotType] = useState(null);
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
      setSelectedPlotType(plotType);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  // Handle button clicks to fetch specific visualizations
  const handleFetchVisualization = (plotType) => {
    fetchVisualizations(plotType);
  };

  // Map plot types to human-readable titles
  const plotTitles = {
    education: 'Education Level',
    contraceptive: 'Contraceptive Use',
    income: 'Family Income',
    healthcare: 'Healthcare Access',
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
            {loading && selectedPlotType === 'education' ? 'Loading...' : 'Education Level'}
          </button>
          <button
            onClick={() => handleFetchVisualization('contraceptive')}
            disabled={loading}
          >
            {loading && selectedPlotType === 'contraceptive' ? 'Loading...' : 'Contraceptive Use'}
          </button>
          <button
            onClick={() => handleFetchVisualization('income')}
            disabled={loading}
          >
            {loading && selectedPlotType === 'income' ? 'Loading...' : 'Family Income'}
          </button>
          <button
            onClick={() => handleFetchVisualization('healthcare')}
            disabled={loading}
          >
            {loading && selectedPlotType === 'healthcare' ? 'Loading...' : 'Healthcare Access'}
          </button>
        </div>

        {error && <p className="error-message">{error}</p>}

        <div className="visualizations-grid">
          {selectedPlotType ? (
            <div className="visualization-section">
              <h3>{plotTitles[selectedPlotType]}</h3>
              <div className="plots-container">
                {visualizations[selectedPlotType].plot1 ? (
                  <img
                    src={visualizations[selectedPlotType].plot1}
                    alt={`${plotTitles[selectedPlotType]} Plot 1`}
                    className="plot-image"
                  />
                ) : (
                  <p>No visualization available yet. Please wait...</p>
                )}
                {visualizations[selectedPlotType].plot2 ? (
                  <img
                    src={visualizations[selectedPlotType].plot2}
                    alt={`${plotTitles[selectedPlotType]} Plot 2`}
                    className="plot-image"
                  />
                ) : null}
              </div>
            </div>
          ) : (
            <p>Select a visualization type to view the data.</p>
          )}
        </div>
      </div>
    </div>
  );
};

export default VisualizationsPage;