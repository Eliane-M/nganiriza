// Retraining.jsx
import React, { useState } from 'react';
import '../assets/css/retraining/retrainingPage.css';

const Retraining = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [isRetraining, setIsRetraining] = useState(false);
  const [uploadMessage, setUploadMessage] = useState('');

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    setSelectedFile(file);
    setUploadMessage('');
    if (file) {
      console.log('File selected for retraining:', file.name);
    }
  };

  const handleRetrain = async () => {
    if (!selectedFile) {
      alert('Please upload a file before retraining.');
      return;
    }

    setIsRetraining(true);
    setUploadMessage('Uploading file and retraining the model...');

    const formData = new FormData();
    formData.append("x_file", selectedFile);
    formData.append("replace_existing", true);

    try {
      const response = await fetch("http://localhost:8000/retrain", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();
      if (response.ok) {
        setUploadMessage(`Retraining completed: ${data.records_inserted} records added.`);
      } else {
        throw new Error(data.detail || "Upload failed");
      }
    } catch (error) {
      setUploadMessage(`Error: ${error.message}`);
    } finally {
      setIsRetraining(false);
    }
  };

  return (
    <div className="retraining-container">
      <header className="header">
        <h1><a href='/'>NGANIRIZA</a></h1>
        <nav>
          <a href="/user-info">Predictions</a>
          <a href="/visualizations">Visualizations</a>
          <a href="/data-upload">Data Upload</a>
          <a href="/retraining">Retraining</a>
        </nav>
      </header>

      <main className="main-content">
        <h2>Upload Dataset to Retrain</h2>
        <p>Improve predictions by retraining the model with new data.</p>

        <div className="retraining-section">
          <div className="upload-area">
            <label htmlFor="file-upload" className="upload-button">
              Upload New Data
            </label>
            <input
              id="file-upload"
              type="file"
              accept=".csv, .xlsx, .json"
              onChange={handleFileChange}
              style={{ display: 'none' }}
            />
            {selectedFile && (
              <p className="file-info">Selected File: {selectedFile.name}</p>
            )}
          </div>

          <button
            className="retrain-button"
            onClick={handleRetrain}
            disabled={isRetraining}
          >
            {isRetraining ? 'Retraining...' : 'Start Retraining'}
          </button>
          {uploadMessage && <p className="upload-message">{uploadMessage}</p>}
        </div>
      </main>
    </div>
  );
};

export default Retraining;
