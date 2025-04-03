import logo from './logo.svg';
import './App.css';
import LandingPage from './pages/landingPage';
import UserInfoPage from './pages/userInfoPage';
import DataUpload from './pages/dataUploadPage';
import Retraining from './pages/retrainingPage';
import VisualizationsPage from './pages/visualizationsPage';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<LandingPage />} />
        <Route path="/user-info" element={<UserInfoPage />} />
        <Route path="/data-upload" element={<DataUpload />} />
        <Route path="/retraining" element={<Retraining />} />
        <Route path="/visualizations" element={<VisualizationsPage />} />
      </Routes>
    </Router>
  );
}

export default App;
