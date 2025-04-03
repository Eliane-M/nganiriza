// DataUpload.jsx
import React, { useState } from 'react';
import '../assets/css/dataUpload/dataUpload.css';
import { Link } from 'react-router-dom';

const DataUpload = () => {
  const [selectedFile, setSelectedFile] = useState(null);

  const handleFileUpload = async () => {
    if (!selectedFile) {
      console.log("No file selected");
      return;
    }
  
    const formData = new FormData();
    formData.append("x_file", selectedFile);
    formData.append("replace_existing", false);
  
    try {
      const response = await fetch("http://localhost:8000/upload", {
        method: "POST",
        body: formData,
      });
  
      const data = await response.json();
      console.log("Response:", data);
    } catch (error) {
      console.error("Error uploading file:", error);
    }
  };
  

  const handleFileChange = async (event) => {
    const file = event.target.files[0];
    setSelectedFile(file);
  
    if (file) {
      console.log("File selected:", file.name);
  
      const formData = new FormData();
      formData.append("file", file);
  
      try {
        const response = await fetch("http://localhost:8000/upload", {
          method: "POST",
          body: formData, // Send form data
        });
  
        const data = await response.json();
        console.log("Upload response:", data);
        
        if (response.ok) {
          alert("File uploaded successfully!");
        } else {
          alert(`Upload failed: ${data.detail}`);
        }
      } catch (error) {
        console.error("Error uploading file:", error);
        alert("An error occurred while uploading the file.");
      }
    }
  };
  

  return (
    <div className="data-upload-container">
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
        <h2>Upload Data For Retraining</h2>
        <p>Securely upload your data to get personalized health insights.</p>

        <div className="upload-section">
          <label htmlFor="file-upload" className="upload-button">
            Choose File
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
      </main>
    </div>
  );
};

export default DataUpload;