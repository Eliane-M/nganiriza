import React, { useState } from 'react';
import '../assets/css/userInfo/userInfo.css';
import { Link } from 'react-router-dom';

const UserInfoPage = () => {
  // State to manage form data
  const [formData, setFormData] = useState({
    age: '',
    educationLevel: '',
    ubudeheCategory: '',
    familyIncome: '',
    healthcareAccessScore: '',
    sexualEducationHours: '',
    contraceptiveUse: '',
    peerInfluence: '',
    parentalInvolvement: '',
    communityResources: '',
    riskCategory: '',
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
      ">6000000": [6000001, 10000000], // Assume max 10,000,000 for upper range
    };

    const [min, max] = incomeRanges[range] || [0, 0];
    return Math.floor(Math.random() * (max - min + 1)) + min;
  };

  // Handle input changes
  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData((prevData) => ({
      ...prevData,
      [name]: value,
    }));
  };

  // Handle form submission
  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      const response = await fetch('http://localhost:8000/predict', {  // Update with your backend URL
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ features: formData }),
        familyIncome: Number(getRandomIncome(formData.familyIncome)),
      });

      if (!response.ok) {
        console.log('Repsonse:', response);
        throw new Error('Failed to get prediction');
      }

      const data = await response.json();
      setPrediction(data.prediction); // Assuming the API response is { "prediction": "some_value" }
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
          {/* <Link to="/user-info" className="nav-link">About</Link> */}
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
            <label htmlFor="age">Age</label>
            <input
              type="number"
              id="age"
              name="age"
              min='5'
              value={formData.age}
              onChange={handleChange}
              placeholder="Enter your age"
              required
            />
          </div>

          <div className="form-group">
            <label htmlFor="educationLevel">Education Level</label>
            <select
              id="educationLevel"
              name="educationLevel"
              value={formData.educationLevel}
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
            <label htmlFor="ubudeheCategory">Ubudehe Category</label>
            <select
              id="ubudeheCategory"
              name="ubudeheCategory"
              value={formData.ubudeheCategory}
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
            <label htmlFor="familyIncome">Family Income (RWF per year)</label>
            <select
              id="familyIncome"
              name="familyIncome"
              value={formData.familyIncome}
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
          </div>

          <div className="form-group">
            <label htmlFor="healthcareAccessScore">Healthcare Access Score (1-100)</label>
            <input
              type="number"
              id="healthcareAccessScore"
              name="healthcareAccessScore"
              value={formData.healthcareAccessScore}
              onChange={handleChange}
              placeholder="Rate from 1 to 100"
              min="1"
              max="100"
              required
            />
          </div>

          <div className="form-group">
            <label htmlFor="sexualEducationHours">Sexual Education Hours Per Week</label>
            <input
              type="number"
              id="sexualEducationHours"
              name="sexualEducationHours"
              value={formData.sexualEducationHours}
              onChange={handleChange}
              min='0'
              max='168'
              placeholder="Hours of sexual education per week"
              required
            />
          </div>

          <div className="form-group">
            <label htmlFor="contraceptiveUse">Contraceptive Use</label>
            <select
              id="contraceptiveUse"
              name="contraceptiveUse"
              value={formData.contraceptiveUse}
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
            <label htmlFor="peerInfluence">Peer Influence (1-100)</label>
            <input
              type="number"
              id="peerInfluence"
              name="peerInfluence"
              value={formData.peerInfluence}
              onChange={handleChange}
              placeholder="Rate from 1 to 100"
              min="1"
              max="100"
              required
            />
          </div>

          <div className="form-group">
            <label htmlFor="parentalInvolvement">Parental Involvement (1-100)</label>
            <input
              type="number"
              id="parentalInvolvement"
              name="parentalInvolvement"
              value={formData.parentalInvolvement}
              onChange={handleChange}
              placeholder="Rate from 1 to 100"
              min="1"
              max="100"
              required
            />
          </div>

          <div className="form-group">
            <label htmlFor="communityResources">Community Resources (1-100)</label>
            <input
              type="number"
              id="communityResources"
              name="communityResources"
              value={formData.communityResources}
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
          <div className="prediction-result">
            <h3>Prediction Result:</h3>
            <p>{prediction}</p>
          </div>
        )}

        {error && <p className="error-message">{error}</p>}
      </div>
    </div>
  );
};

export default UserInfoPage;