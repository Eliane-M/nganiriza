import React, { useState } from 'react';
import '../assets/css/userInfo/userInfo.css';
import { Link } from 'react-router-dom';

const UserInfoPage = () => {
  const [formData, setFormData] = useState({
    Age: '',
    Education_Level: '',
    Ubudehe_Category: '',
    Family_Income_Range: '',
    Family_Income_Rwf: '',
    Healthcare_Access_Score: '',
    Sexual_Education_Hours: '',
    Contraceptive_Use: '',
    Peer_Influence: '',
    Parental_Involvement: '',
    Community_Resources: '',
  });

  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Function to generate a random number within a given range
  const getRandomIncome = (range) => {
    const incomeRanges = {
      "<1200000": [0, 1199999],
      "1200000-3000000": [1200000, 3000000],
      "3000000-6000000": [3000000, 6000000],
      ">6000000": [6000001, 10000000],
    };

    const [min, max] = incomeRanges[range] || [0, 0];
    return Math.floor(Math.random() * (max - min + 1)) + min;
  };

  // Handle input changes
  const handleChange = (e) => {
    const { name, value } = e.target;

    if (name === "Family_Income_Range") {
      // When the user selects an income range, generate a random income and store both the range and the number
      const randomIncome = getRandomIncome(value);
      setFormData((prevData) => ({
        ...prevData,
        Family_Income_Range: value, // Store the selected range
        Family_Income_Rwf: randomIncome, // Store the random number
      }));
    } else {
      setFormData((prevData) => ({
        ...prevData,
        [name]: value,
      }));
    }
  };

  // Handle form submission
  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    // Prepare the data in the format expected by the backend
    const payload = {
      Age: parseInt(formData.Age),
      Education_Level: formData.Education_Level,
      Ubudehe_Category: formData.Ubudehe_Category,
      Family_Income_Rwf: parseFloat(formData.Family_Income_Rwf), // Use the stored random number
      Healthcare_Access_Score: parseInt(formData.Healthcare_Access_Score),
      Sexual_Education_Hours: parseFloat(formData.Sexual_Education_Hours),
      Contraceptive_Use: formData.Contraceptive_Use,
      Peer_Influence: parseInt(formData.Peer_Influence),
      Parental_Involvement: parseInt(formData.Parental_Involvement),
      Community_Resources: parseInt(formData.Community_Resources),
    };

    try {
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to get prediction');
      }

      const data = await response.json();
      setPrediction(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const incomeRanges = [
    { label: "Less than 1,200,000 RWF per year", value: "<1200000" },
    { label: "1,200,000 - 3,000,000 RWF per year", value: "1200000-3000000" },
    { label: "3,000,000 - 6,000,000 RWF per year", value: "3000000-6000000" },
    { label: "More than 6,000,000 RWF per year", value: ">6000000" },
  ];

  return (
    <div className="user-info-form-container">
      <header className="header">
        <h1><a href='/'>NGANIRIZA</a></h1>
        <nav>
          <Link to="/user-info" className="nav-link">Predictions</Link>
          <Link to="/visualizations" className="nav-link">Visualizations</Link>
          <Link to="/data-upload" className="nav-link">Data Upload</Link>
          <Link to="/retraining" className="nav-link">Retrain</Link>
        </nav>
      </header>

      <div className="form-wrapper">
        <h2 className='title'>Tell Us About Yourself</h2>
        <p>Let's get to know you better to provide the best support for your journey.</p>

        <form onSubmit={handleSubmit} className="user-info-form">
          <div className="form-group">
            <label htmlFor="Age">Age</label>
            <input
              type="number"
              id="Age"
              name="Age"
              min='5'
              value={formData.Age}
              onChange={handleChange}
              placeholder="Enter your age"
              required
            />
          </div>

          <div className="form-group">
            <label htmlFor="Education_Level">Education Level</label>
            <select
              id="Education_Level"
              name="Education_Level"
              value={formData.Education_Level}
              onChange={handleChange}
              required
            >
              <option value="">Select your education level</option>
              <option value="primary">Primary</option>
              <option value="secondary">O' Level</option>
              <option value="tertiary">A' Level</option>
              <option value="none">None</option>
            </select>
          </div>

          <div className="form-group">
            <label htmlFor="Ubudehe_Category">Ubudehe Category</label>
            <select
              id="Ubudehe_Category"
              name="Ubudehe_Category"
              value={formData.Ubudehe_Category}
              onChange={handleChange}
              required
            >
              <option value="">Select your category</option>
              <option value="1">A</option>
              <option value="2">B</option>
              <option value="3">C</option>
              <option value="4">D</option>
              <option value="5">E</option>
            </select>
          </div>

          <div className="form-group">
            <label htmlFor="Family_Income_Range">Family Income (RWF per year)</label>
            <select
              id="Family_Income_Range"
              name="Family_Income_Range"
              value={formData.Family_Income_Range} // Use the stored range
              onChange={handleChange}
              required
            >
              <option value="">Select income range</option>
              {incomeRanges.map((range) => (
                <option key={range.value} value={range.value}>
                  {range.label}
                </option>
              ))}
            </select>
            {/* {formData.Family_Income_Rwf && (
              <p>Estimated Income: {formData.Family_Income_Rwf.toLocaleString()} RWF</p>
            )} */}
          </div>

          <div className="form-group">
            <label htmlFor="Healthcare_Access_Score">Healthcare Access Score (1-100)</label>
            <input
              type="number"
              id="Healthcare_Access_Score"
              name="Healthcare_Access_Score"
              value={formData.Healthcare_Access_Score}
              onChange={handleChange}
              placeholder="Rate from 1 to 100"
              min="1"
              max="100"
              required
            />
          </div>

          <div className="form-group">
            <label htmlFor="Sexual_Education_Hours">Sexual Education Hours Per Week</label>
            <input
              type="float"
              id="Sexual_Education_Hours"
              name="Sexual_Education_Hours"
              value={formData.Sexual_Education_Hours}
              onChange={handleChange}
              min='0'
              max='10'
              placeholder="Hours of sexual education per week on average"
              required
            />
          </div>

          <div className="form-group">
            <label htmlFor="Contraceptive_Use">Contraceptive Use</label>
            <select
              id="Contraceptive_Use"
              name="Contraceptive_Use"
              value={formData.Contraceptive_Use}
              onChange={handleChange}
              required
            >
              <option value="">Select an option</option>
              <option value="yes">Never</option>
              <option value="no">Regular</option>
              <option value="inconsistent">Inconsistent</option>
            </select>
          </div>

          <div className="form-group">
            <label htmlFor="Peer_Influence">Peer Influence (1-100)</label>
            <input
              type="number"
              id="Peer_Influence"
              name="Peer_Influence"
              value={formData.Peer_Influence}
              onChange={handleChange}
              placeholder="Rate from 1 to 100"
              min="1"
              max="100"
              required
            />
          </div>

          <div className="form-group">
            <label htmlFor="Parental_Involvement">Parental Involvement (1-100)</label>
            <input
              type="number"
              id="Parental_Involvement"
              name="Parental_Involvement"
              value={formData.Parental_Involvement}
              onChange={handleChange}
              placeholder="Rate from 1 to 100"
              min="1"
              max="100"
              required
            />
          </div>

          <div className="form-group">
            <label htmlFor="Community_Resources">Community Resources (1-100)</label>
            <input
              type="number"
              id="Community_Resources"
              name="Community_Resources"
              value={formData.Community_Resources}
              onChange={handleChange}
              placeholder="Rate from 1 to 100 the available community resources"
              min="1"
              max="100"
              required
            />
          </div>

          <button type="submit" className="submit-btn">
            {loading ? 'Submitting...' : 'Submit'}
          </button>
        </form>

        {prediction && (
          <div className="prediction-result" style={{
            backgroundColor: prediction['Risk_Category'] === 'High Risk' 
              ? 'rgba(255, 0, 0, 0.42)'
              : prediction['Risk_Category'] === 'Medium Risk' 
              ? 'rgba(255, 166, 0, 0.39)'
              : 'rgba(0, 128, 0, 0.4)',
            padding: '10px',
            marginTop: '20px',
            borderRadius: '25px',
          }}>
            <h3>Prediction Result:</h3>
            <p>You are at a {prediction['Risk_Category']}</p>
          </div>
        )}

        {error && <p className="error-message">{error}</p>}
      </div>
    </div>
  );
};

export default UserInfoPage;