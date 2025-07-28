// App.jsx
import { Routes, Route, Navigate } from 'react-router-dom';
import useAuth from './context/useAuth';

import Home from './pages/Home';
import Upload from './pages/Upload';
import Ask from './pages/Ask';
import Documents from './pages/Documents';
import Profile from './pages/Profile';
import LoginPage from './pages/Login';
import RegisterPage from './pages/Register';
import Layout from './components//ui/Layout';

function PrivateRoute({ children }) {
  const { user } = useAuth();
  return user ? children : <Navigate to="/login" />;
}

function App() {
  return (
    <Routes>
      <Route path="/login" element={<LoginPage />} />
      <Route path="/register" element={<RegisterPage />} />

      {/* Private layout chứa navbar và các trang cần đăng nhập */}
      <Route
        path="/"
        element={
          <PrivateRoute>
            <Layout />
          </PrivateRoute>
        }
      >
        <Route index element={<Home />} />
        <Route path="upload" element={<Upload />} />
        <Route path="ask" element={<Ask />} />
        <Route path="documents" element={<Documents />} />
        <Route path="profile" element={<Profile />} />
      </Route>
    </Routes>
  );
}

export default App;
